#include "base/pull_serializer.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <list>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "base/array.hpp"
#include "base/not_null.hpp"
#include "base/sink_source.hpp"
#include "gipfeli/compression.h"
#include "gipfeli/gipfeli.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "serialization/geometry.pb.h"
#include "serialization/physics.pb.h"
#include "serialization/quantities.pb.h"

namespace principia {
namespace base {

using serialization::DiscreteTrajectory;
using serialization::Pair;
using serialization::Point;
using serialization::Quantity;
using ::std::placeholders::_1;
using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using namespace principia::base::_array;
using namespace principia::base::_not_null;
using namespace principia::base::_pull_serializer;
using namespace principia::base::_sink_source;

namespace this_internal = _pull_serializer::internal;

namespace {
  int const chunk_size = 99;
  int const number_of_chunks = 3;
  int const runs_per_test = 1000;
  int const small_chunk_size = 3;
}  // namespace

class PullSerializerTest : public ::testing::Test {
 protected:
  PullSerializerTest()
      : pull_serializer_(
            std::make_unique<PullSerializer>(chunk_size,
                                             number_of_chunks,
                                             /*compressor=*/nullptr)),
        stream_(Array<std::uint8_t>(data_, small_chunk_size),
                std::bind(&PullSerializerTest::OnFull,
                          this,
                          _1,
                          std::ref(strings_))) {}

  static not_null<std::unique_ptr<DiscreteTrajectory const>> BuildTrajectory() {
    not_null<std::unique_ptr<DiscreteTrajectory>> result =
        make_not_null_unique<DiscreteTrajectory>();
    // Build a biggish protobuf for serialization.
    for (int i = 0; i < 100; ++i) {
      auto* const idof = result->add_timeline();
      Point* instant = idof->mutable_instant();
      Quantity* scalar = instant->mutable_scalar();
      scalar->set_dimensions(3);
      scalar->set_magnitude(3 * i);
      Pair* dof = idof->mutable_degrees_of_freedom();
      Pair::Element* t1 = dof->mutable_t1();
      Point* point1 = t1->mutable_point();
      Quantity* scalar1 = point1->mutable_scalar();
      scalar1->set_dimensions(1);
      scalar1->set_magnitude(i);
      Pair::Element* t2 = dof->mutable_t2();
      Point* point2 = t2->mutable_point();
      Quantity* scalar2 = point2->mutable_scalar();
      scalar2->set_dimensions(2);
      scalar2->set_magnitude(2 * i);
    }
    return std::move(result);
  }

  // Returns the first string in the list.  Note that the very first string is
  // always discarded.
  Array<std::uint8_t> OnFull(Array<std::uint8_t> const bytes,
                             std::list<std::string>& strings) {
    strings.emplace_back(reinterpret_cast<const char*>(&bytes.data[0]),
                         static_cast<std::size_t>(bytes.size));
    return Array<std::uint8_t>(data_, small_chunk_size);
  }

  std::unique_ptr<PullSerializer> pull_serializer_;
  this_internal::DelegatingArrayOutputStream stream_;
  std::list<std::string> strings_;
  std::uint8_t data_[small_chunk_size];
};

TEST_F(PullSerializerTest, Stream) {
  void* data;
  int size;

  EXPECT_TRUE(stream_.Next(&data, &size));
  EXPECT_EQ(3, size);
  EXPECT_EQ(3, stream_.ByteCount());
  std::memcpy(data, "abc", 3);
  EXPECT_TRUE(stream_.Next(&data, &size));
  EXPECT_EQ(3, size);
  EXPECT_EQ(6, stream_.ByteCount());
  std::memcpy(data, "xy", 2);
  stream_.BackUp(1);
  EXPECT_EQ(5, stream_.ByteCount());
  EXPECT_TRUE(stream_.Next(&data, &size));
  EXPECT_EQ(3, size);
  EXPECT_EQ(8, stream_.ByteCount());
  std::memcpy(data, "uvw", 3);
  stream_.BackUp(2);
  EXPECT_EQ(6, stream_.ByteCount());
  EXPECT_THAT(strings_, ElementsAre("abc", "xy", "u"));
}

TEST_F(PullSerializerTest, SerializationSizes) {
  auto trajectory = BuildTrajectory();
  pull_serializer_->Start(std::move(trajectory));
  std::vector<std::int64_t> actual_sizes;
  std::vector<std::int64_t> expected_sizes(53, chunk_size);
  expected_sizes.push_back(53);
  for (;;) {
    Array<std::uint8_t> const bytes = pull_serializer_->Pull();
    if (bytes.size == 0) {
      break;
    }
    actual_sizes.push_back(bytes.size);
  }
  EXPECT_THAT(actual_sizes, ElementsAreArray(expected_sizes));
}

TEST_F(PullSerializerTest, SerializationGipfeli) {
  std::string uncompressed1;
  std::string uncompressed2;
  {
    auto trajectory = BuildTrajectory();
    pull_serializer_->Start(std::move(trajectory));
    for (;;) {
      Array<std::uint8_t> const bytes = pull_serializer_->Pull();
      if (bytes.size == 0) {
        break;
      }
      for (int i = 0; i < bytes.size; ++i) {
        uncompressed1.append(1, bytes.data[i]);
      }
    }
  }
  {
    auto const compressed_pull_serializer =
        std::make_unique<PullSerializer>(
            chunk_size,
            /*number_of_chunks=*/4,
            google::compression::NewGipfeliCompressor());
    auto trajectory = BuildTrajectory();
    compressed_pull_serializer->Start(std::move(trajectory));
    auto compressor = google::compression::NewGipfeliCompressor();
    for (;;) {
      Array<std::uint8_t> const bytes = compressed_pull_serializer->Pull();
      if (bytes.size == 0) {
        break;
      }
      std::string compressed;
      std::string uncompressed;
      for (int i = 0; i < bytes.size; ++i) {
        compressed.append(1, bytes.data[i]);
      }
      compressor->Uncompress(compressed, &uncompressed);
      uncompressed2.append(uncompressed);
    }
  }

  EXPECT_EQ(uncompressed1, uncompressed2);
}

TEST_F(PullSerializerTest, SerializationThreading) {
  DiscreteTrajectory read_trajectory;
  auto const trajectory = BuildTrajectory();
  int const byte_size = trajectory->ByteSize();
  auto expected_serialized_trajectory =
      std::make_unique<std::uint8_t[]>(byte_size);
  CHECK(trajectory->SerializePartialToArray(&expected_serialized_trajectory[0],
                                            byte_size));

  // Run this test repeatedly to detect threading issues (it will flake in case
  // of problems).
  for (int i = 0; i < runs_per_test; ++i) {
    auto trajectory = BuildTrajectory();
    auto actual_serialized_trajectory =
        std::make_unique<std::uint8_t[]>(byte_size);
    std::uint8_t* data = &actual_serialized_trajectory[0];

    // The serialization happens concurrently with the test.
    pull_serializer_ = std::make_unique<PullSerializer>(chunk_size,
                                                        number_of_chunks,
                                                        /*compressor=*/nullptr);
    pull_serializer_->Start(std::move(trajectory));
    for (;;) {
      Array<std::uint8_t> const bytes = pull_serializer_->Pull();
      std::memcpy(data, bytes.data, static_cast<std::size_t>(bytes.size));
      data = &data[bytes.size];
      if (bytes.size == 0) {
        break;
      }
    }
    pull_serializer_.reset();

    // Check if the serialized version can be parsed and if not print the first
    // difference.
    if (!read_trajectory.ParseFromArray(&actual_serialized_trajectory[0],
                                        byte_size)) {
      for (int i = 0; i < byte_size; ++i) {
        if (expected_serialized_trajectory[i] !=
            actual_serialized_trajectory[i]) {
          LOG(FATAL) << "position=" << i
                     << ", expected="
                     << static_cast<int>(expected_serialized_trajectory[i])
                     << ", actual="
                     << static_cast<int>(actual_serialized_trajectory[i]);
        }
      }
    }
  }
}

TEST(ArraySinkTest, GetAppendBufferFallsBackToScratch) {
  // The compressor may request more room than the array has left even though
  // its actual output fits; the `Sink` contract then lets us return the
  // caller-owned scratch buffer (seen in the field: 82050 vs. 82049).
  std::uint8_t data[16];
  ArraySink<std::uint8_t> sink(Array<std::uint8_t>(data, 16));
  sink.Append("0123456789", 10);

  // A request that fits — exactly — stays on the zero-copy path.
  char scratch[32];
  std::size_t allocated_size;
  char* buffer = sink.GetAppendBuffer(/*min_size=*/6,
                                      /*desired_size_hint=*/32,
                                      scratch,
                                      /*scratch_size=*/32,
                                      &allocated_size);
  EXPECT_EQ(reinterpret_cast<char*>(&data[10]), buffer);
  EXPECT_EQ(6, allocated_size);

  buffer = sink.GetAppendBuffer(/*min_size=*/7,
                                /*desired_size_hint=*/7,
                                scratch,
                                /*scratch_size=*/32,
                                &allocated_size);
  EXPECT_EQ(scratch, buffer);
  EXPECT_GE(allocated_size, 7);

  // The actual output is smaller than the request and must still fit.
  std::memcpy(buffer, "abcdef", 6);
  sink.Append(buffer, 6);
  EXPECT_EQ(16, sink.array().size);
  EXPECT_EQ(0, std::memcmp(data, "0123456789abcdef", 16));
}

TEST(ArraySinkTest, CompressorOverRequestGoesThroughScratch) {
  // Compress a compressible block once to learn its actual output size, then
  // again into an array with exactly that much room: the compressor's
  // conservative request no longer fits and must take the scratch path —
  // exercising the `AppendMemBlock` handoff — while producing the same bytes.
  std::vector<std::uint8_t> input;
  std::uint8_t filler = 0;
  while (input.size() < std::size_t{64 << 10}) {
    for (std::uint8_t const c : {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'}) {
      input.push_back(c);
    }
    input.push_back(++filler);
  }
  input.resize(std::size_t{64 << 10});
  auto const compressor = google::compression::NewGipfeliCompressor();
  auto const compress =
      [&compressor, &input](Array<std::uint8_t> const array) {
    ArraySource<std::uint8_t> source(
        Array<std::uint8_t>(input.data(), input.size()));
    ArraySink<std::uint8_t> sink(array);
    compressor->CompressStream(&source, &sink);
    return sink.array().size;
  };

  UniqueArray<std::uint8_t> const generous(2 * input.size());
  std::int64_t const actual_size = compress(generous.get());
  EXPECT_LT(0, actual_size);
  EXPECT_LT(actual_size, static_cast<std::int64_t>(input.size()));

  UniqueArray<std::uint8_t> const tight(actual_size);
  EXPECT_EQ(actual_size, compress(tight.get()));
  EXPECT_EQ(0,
            std::memcmp(generous.data.get(), tight.data.get(), actual_size));
}

}  // namespace base
}  // namespace principia
