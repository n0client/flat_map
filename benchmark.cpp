
#include <cstdint>
#include <chrono>

#include "flat_map.h"

#define BOUND 100000000

// yes, i know i should use random numbers - i will do this later
// this was to check correctness

int main()
{
  flat_map<uint64_t, uint64_t> map;

  auto start_time = std::chrono::steady_clock::now();
  for (uint64_t i = 0; i < BOUND; ++i)
  {
    map.insert({i, i});
  }
  auto end_time = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
  std::cout << "inserted " << BOUND << " uint64_t pairs in " << duration.count() << " ms\n";


  start_time = std::chrono::steady_clock::now();
  uint64_t j = 0;
  for (uint64_t i = 0; i < BOUND; ++i)
  {
    j += map.contains(i);
  }
  end_time = std::chrono::steady_clock::now();
  auto d2 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
  std::cout << "looked up " << BOUND << " uint64_t pairs in " << d2.count() << " ms\n";
  std::cout << "checksum " << j << "\n";


  start_time = std::chrono::steady_clock::now();
  j = 0;
  for (uint64_t i = 0; i < BOUND; ++i)
  {
    j += map.remove(i);
  }
  end_time = std::chrono::steady_clock::now();
  d2 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
  std::cout << "removed " << BOUND << " uint64_t pairs in " << d2.count() << " ms\n";
  std::cout << "Checksum " << j << "\n";

  start_time = std::chrono::steady_clock::now();
  map.clear();
  end_time = std::chrono::steady_clock::now();
  d2 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
  std::cout << "cleared " << BOUND << " uint64_t pairs in " << d2.count() << " ms\n";
}
