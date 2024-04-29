#include <atomic>
#include <dragonbox/dragonbox.h>
#include <gtest/gtest.h>
#define FMT_USE_FULL_CACHE_DRAGONBOX 1
#define FMT_HEADER_ONLY 1
#include <fmt/format.h>
#include <fmt/compile.h>
#include <dragonbox/dragonbox_to_chars.h>
#include <random>

class PerformanceTest : public ::testing::Test {
protected:
    void SetUp() override {
        std::seed_seq seed{42};
        std::mt19937 gen(seed);
        std::uniform_real_distribution<double> dis_double(-100000.0, +100000.0);
        std::uniform_real_distribution<float> dis_float(-100000.0, +100000.0);
        
        for (int i = 0; i < 100000000; ++i) {
            values_double.push_back(dis_double(gen));
            values_float.push_back(dis_float(gen));
        }
    }

    void TearDown() override {
    }

    std::vector<double> values_double;
    std::vector<float> values_float;
};

TEST_F(PerformanceTest, FmtPerformanceDouble) {
    char buffer[20];

    for (const auto& value : values_double) {
        fmt::format_to(buffer, FMT_COMPILE("{}"), value);
    }
}

TEST_F(PerformanceTest, DragonboxPerformanceDouble) {
    char buffer[20];

    for (const auto& value : values_double) {
        jkj::dragonbox::to_chars(value, buffer, jkj::dragonbox::policy::cache::full);
    }
}


TEST_F(PerformanceTest, FmtPerformanceFloat) {
    char buffer[20];

    for (const auto& value : values_float) {
        fmt::format_to(buffer, FMT_COMPILE("{}"), value);
    }
}

TEST_F(PerformanceTest, DragonboxPerformanceFloat) {
    char buffer[20];

    for (const auto& value : values_double) {
        jkj::dragonbox::to_chars(value, buffer, jkj::dragonbox::policy::cache::full);
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}