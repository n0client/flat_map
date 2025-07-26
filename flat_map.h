
#pragma once

#include <arm_neon.h>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <type_traits>
#include <utility>

#define BLOCK_SIZE 16
#define TAG_SHIFT 56

template <typename Key, typename Value>
class flat_map
{
public:
  uint64_t cur_size, max_size;
  uint8_t *tags, *dists; /* combine dists + tags, too much mem */
  struct kv_pair
  {
    Key k;
    Value v;
    kv_pair(Key _k, Value _v) : k(_k), v(_v) {}
  } *data;

  inline void init()
  {
    tags = (uint8_t *)calloc(max_size, sizeof(uint8_t));
    dists = (uint8_t *)calloc(max_size, sizeof(uint8_t));
    data = (kv_pair *)calloc(max_size, sizeof(kv_pair));
  }

  uint64_t splittable64(uint64_t x) const
{
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ull;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebull;
    x ^= x >> 31;
    return x | (1ull << TAG_SHIFT);
}

  /* optimize */
  static inline uint16_t uint8x16_to_mask(uint8x16_t vec)
  {
    constexpr uint8x16_t mask = {1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};
    uint8x16_t res = vandq_u8(vec, mask);
    uint16_t lo  = vaddlv_u8(vget_low_u8(res));
    uint16_t hi  = vaddlv_u8(vget_high_u8(res));
    return lo | hi << 8;  
  }

  uint64_t _find(const Key &k) const
  {
    uint64_t h = splittable64(k);
    uint8_t tag = h >> TAG_SHIFT, dist = 0;
    uint64_t iter = __builtin_ctz(max_size) * 3;
    for (uint64_t i = 0; i < iter; ++i)
    {
      uint64_t index = (h + i * BLOCK_SIZE) & (max_size - 1) & ~(BLOCK_SIZE - 1);
      uint8x16_t dtag_vec = vld1q_u8(tags + index);
      uint8x16_t tag_vec  = vdupq_n_u8(tag);
      uint16_t mask = uint8x16_to_mask(vceqq_u8(dtag_vec, tag_vec));

      while (mask)
      {
        uint8_t offset = __builtin_ctz(mask);
        mask &= ~(1 << offset);

        if (k == data[index + offset].k)
          return index + offset;
      }

      uint8x16_t ddist_vec = vld1q_u8(dists + index);
      uint8x16_t dist_vec  = vdupq_n_u8(dist);
      uint8_t continue_search = vmaxvq_u8(vcgeq_u8(ddist_vec, dist_vec));

      if (!continue_search)
        return UINT64_MAX;

      dist++;
    }
    return UINT64_MAX; // kind of an eh design choice
  }


  template <bool caller_is_resize = false>
  Value *_insert(kv_pair &&p, kv_pair *in_data, uint8_t *in_tags, uint8_t *in_dist, uint64_t in_size)
  {
    uint64_t h = splittable64(p.k);
    uint8_t tag = h >> TAG_SHIFT;
    uint8_t dist = 0;

    //uint64_t iter = 108;
    //uint64_t iter = 96;
    uint64_t iter = __builtin_ctz(in_size) * 3;
    uint64_t stop_here = h & (in_size - 1) & ~(BLOCK_SIZE - 1), index = 2;
    //uint64_t iter = __builtin_ctz(in_size) * 4;
    for (uint64_t i = 0; i < iter /* && i && stop_here != index */; ++i)
    {
      index = (h + i * BLOCK_SIZE) & (in_size - 1) & ~(BLOCK_SIZE - 1);
      uint8x16_t in_tag_vec = vld1q_u8(in_tags + index);
      uint8x16_t tag_vec    = vdupq_n_u8(tag);
      uint8x16_t res_vec    = vceqq_u8(in_tag_vec, tag_vec);
      uint16_t mask = uint8x16_to_mask(res_vec);

      while (mask)
      {
        uint8_t offset = __builtin_ctz(mask);
        mask &= ~(1 << offset);

        if (in_data[index + offset].k == p.k)
        {
          in_data[index + offset].v = std::move(p.v);
          return &in_data[index + offset].v;
        }
      }

      mask = uint8x16_to_mask(vceqzq_u8(in_tag_vec));
      if (mask)
      {
        uint8_t offset = __builtin_ctz(mask);
        in_data[index + offset] = std::move(p);
        in_tags[index + offset] = tag;
        in_dist[index + offset] = dist;
        return &in_data[index + offset].v;
      }

      uint8x16_t in_dist_vec = vld1q_u8(in_dist + index);
      uint8x16_t dist_vec = vdupq_n_u8(dist);
      mask = uint8x16_to_mask(vcltq_u8(in_dist_vec, dist_vec));

      if (mask)
      {
        uint8_t offset = __builtin_ctz(mask);
        std::swap(p, in_data[index + offset]);
        std::swap(tag, in_tags[index + offset]);
        std::swap(dist, in_dist[index + offset]);
      }

      dist += (dist < 255);
    }

    if constexpr (caller_is_resize)
      assert(0);
    else
      return _resize(std::move(p));
  }


  Value *_resize(kv_pair &&p)
  {
    //std::cout << "Resizing at " << (double)cur_size / (double)max_size << "\n";
    uint64_t new_size = max_size << 1;
    uint8_t *new_tags = (uint8_t *)calloc(new_size, sizeof(uint8_t));
    uint8_t *new_dist = (uint8_t *)calloc(new_size, sizeof(uint8_t));
    kv_pair *new_data = (kv_pair *)calloc(new_size, sizeof(kv_pair));

    Value *ret = _insert<true>(std::move(p), new_data, new_tags, new_dist, new_size);

    for (uint64_t i = 0; i < max_size; ++i)
      if (tags[i])
        _insert<true>(std::move(data[i]), new_data, new_tags, new_dist, new_size);

    free(tags);
    free(dists);
    free(data);

    max_size = new_size;
    tags = new_tags;
    dists = new_dist;
    data = new_data;

    return ret;
  }


  flat_map() : cur_size(0), max_size(BLOCK_SIZE) { init(); }
  flat_map(uint64_t size) : cur_size(0), max_size(size) { init(); }
  ~flat_map() 
  { 
    if constexpr (!std::is_trivially_destructible_v<kv_pair>)
    {
      for (uint64_t i = 0; i < max_size; ++i)
        if (tags[i])
          data[i].~kv_pair();
    }

    free(data); 
    free(dists); 
    free(tags); 
  }

  // map.insert({k, p})
  void insert(kv_pair &&_p)
  {
    kv_pair p(std::move(_p));
    if (cur_size >= max_size * 0.9)
      _resize(std::move(p));
    else
      _insert(std::move(p), data, tags, dists, max_size);
    cur_size++;
  }

  Value *find(const Key &k)
  {
    return &data[_find(k)].v;
  }

  const Value *find(const Key &k) const
  {
    return &data[_find(k)].v;
  }

  bool contains(const Key &k) const
  {
    return _find(k) != UINT64_MAX;
  }

  bool remove(const Key &k)
  {
    uint64_t slot = _find(k);

    // actually inline function for slight edge - less branching
    if (slot == UINT64_MAX) return false;

    if constexpr (!std::is_trivially_destructible_v<kv_pair>)
      data[slot].~kv_pair();

    cur_size--;
    tags[slot] = 0;

    return true;

    /* -- backshift algorithm -- requires dist to be set to 0 too
    uint64_t index = empty_slot & ~(BLOCK_SIZE - 1);
    uint64_t next_index = (index + BLOCK_SIZE) & (max_size - 1);
    for (uint64_t i = 0; i < UINT64_MAX; ++i)
    {
      // important to have 0-on-init distance here -- calloc
      uint16_t mask = ~uint8x16_to_mask(vceqzq_u8(vld1q_u8(dists + next_index)));
      if (!mask)
        return true;

      uint8_t offset = __builtin_ctz(mask);
      data[empty_slot] = std::move(data[next_index + offset]);
      assert(tags[empty_slot] == 0);
      tags[empty_slot] = tags[next_index + offset];
      dists[empty_slot] = dists[next_index + offset] - 1;
      tags[next_index + offset] = dists[next_index + offset] = 0;

      index = next_index;
      empty_slot = next_index + offset;
      next_index = (next_index + BLOCK_SIZE) & (max_size - 1);
    }

    return true;
    */
  }

  void clear()
  {
    if constexpr (!std::is_trivially_destructible_v<kv_pair>)
    {
      for (uint64_t i = 0; i < max_size; ++i)
        if (tags[i])
          data[i].~kv_pair();
    }

    for (uint64_t i = 0; i < max_size; ++i)
      tags[i] = 0;
  }
};
