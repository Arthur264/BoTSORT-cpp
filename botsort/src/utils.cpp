#include "utils.h"

#include <iostream>

#include "lapjv.h"

double lapjv(CostMatrix &cost, std::vector<int> &rowsol,
             std::vector<int> &colsol, bool extend_cost, float cost_limit,
             bool return_cost)
{
    int n_rows = static_cast<int>(cost.rows());
    int n_cols = static_cast<int>(cost.cols());
    rowsol.resize(n_rows);
    colsol.resize(n_cols);

    int n = 0;
    if (n_rows == n_cols)
    {
        n = n_rows;
    }
    else
    {
        if (!extend_cost)
        {
            std::cout << "set extend_cost=True" << std::endl;
            exit(0);
        }
    }

    std::vector<std::vector<float>> cost_c;
    if (extend_cost || cost_limit < LONG_MAX)
    {
        n = n_rows + n_cols;
        cost_c.resize(n);
        for (int i = 0; i < cost_c.size(); i++)
            cost_c[i].resize(n);

        if (cost_limit < LONG_MAX)
        {
            for (int i = 0; i < cost_c.size(); i++)
            {
                for (int j = 0; j < cost_c[i].size(); j++)
                {
                    cost_c[i][j] = cost_limit / 2.0;
                }
            }
        }
        else
        {
            float cost_max = -1;
            for (Eigen::Index i = 0; i < cost.rows(); ++i)
            {
                for (Eigen::Index j = 0; j < cost.cols(); ++j)
                {
                    if (cost(i, j) > cost_max)
                        cost_max = cost(i, j);
                }
            }
            for (int i = 0; i < cost_c.size(); i++)
            {
                for (int j = 0; j < cost_c[i].size(); j++)
                {
                    cost_c[i][j] = cost_max + 1;
                }
            }
        }

        for (int i = n_rows; i < cost_c.size(); i++)
        {
            for (int j = n_cols; j < cost_c[i].size(); j++)
            {
                cost_c[i][j] = 0;
            }
        }
        for (Eigen::Index i = 0; i < cost.rows(); ++i)
        {
            for (Eigen::Index j = 0; j < cost.cols(); ++j)
            {
                cost_c[i][j] = cost(i, j);
            }
        }
    }
    else
    {
        cost_c.reserve(cost.rows());
        for (Eigen::Index i = 0; i < cost.rows(); ++i)
        {
            std::vector<float> row;
            for (Eigen::Index j = 0; j < cost.cols(); ++j)
            {
                row.emplace_back(cost(i, j));
            }
            cost_c.emplace_back(row);
        }
    }

    std::vector<int> x_c(n, -1);
    std::vector<int> y_c(n, 0);

    int ret = lapjv_internal(n, cost_c, x_c, y_c);
    if (ret != 0)
    {
        std::cout << "Calculate Wrong!" << std::endl;
        exit(0);
    }

    double opt = 0.0;

    if (n != n_rows)
    {
        for (int i = 0; i < n; i++)
        {
            if (x_c[i] >= n_cols) x_c[i] = -1;
            if (y_c[i] >= n_rows) y_c[i] = -1;
        }
        for (int i = 0; i < n_rows; i++) { rowsol[i] = x_c[i]; }
        for (int i = 0; i < n_cols; i++) { colsol[i] = y_c[i]; }

        if (return_cost)
        {
            for (int i = 0; i < rowsol.size(); i++)
            {
                if (rowsol[i] != -1) { opt += cost_c[i][rowsol[i]]; }
            }
        }
    }
    else if (return_cost)
    {
        for (int i = 0; i < rowsol.size(); i++)
        {
            opt += cost_c[i][rowsol[i]];
        }
    }
    return opt;
}
