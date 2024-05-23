/*
 * @lc app=leetcode.cn id=120 lang=cpp
 * @lcpr version=30202
 *
 * [120] 三角形最小路径和
 */


// @lcpr-template-start
using namespace std;
#include <algorithm>
#include <array>
#include <bitset>
#include <climits>
#include <deque>
#include <functional>
#include <iostream>
#include <list>
#include <queue>
#include <stack>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
// @lcpr-template-end
// @lc code=start
class Solution {
public:
    int minimumTotal(vector<vector<int>>& triangle) {
        int minimum = INT_MAX;
        if (triangle.empty()) {
            return 0;
        }
        int rows = triangle.size();
        int cols = triangle[rows - 1].size();
        for (int i = 0; i < cols; i++) {
            minimum = min(minimum, recursive(triangle, rows - 1, i));
        }
        return minimum;
    }
private:
    int recursive(vector<vector<int>>& triangle, int i, int j) {
        if (i == 0 && j == 0) {
            return triangle[0][0];
        }

        if (i == -1 || j == -1 || i >= triangle.size() || j >= triangle[i].size()) {
            return INT_MAX;
        }

        return min(recursive(triangle, i - 1, j), recursive(triangle, i - 1, j - 1)) + triangle[i][j];
    }
};
// @lc code=end

int main() {
    std::vector<vector<int>> triangle = 
        {{2},{3,4},{6,5,7},{4,1,8,3}};
    Solution s;
    std::cout << s.minimumTotal(triangle) << std::endl;
}

/*
// @lcpr case=start
// [[2],[3,4],[6,5,7],[4,1,8,3]]\n
// @lcpr case=end

// @lcpr case=start
// [[-10]]\n
// @lcpr case=end

 */

