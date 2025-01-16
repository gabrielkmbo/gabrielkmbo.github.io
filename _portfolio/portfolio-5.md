---
title: "Matrix Theory Linear Algebra Algorithms"
excerpt: "In Matrix Analysis for Scientists and Engineers by Alan J. Laub, algorithms are designed to provide efficient computational techniques for linear algebra operations, such as matrix decompositions, solving linear systems, and eigenvalue computations. Building on these concepts, I developed Python implementations of key linear algebra algorithms and matrix formulas foundational to machine learning models. These include Principal Component Analysis (PCA), Singular Value Decomposition (SVD), linear and logistic regression, least squares optimization, and k-means clustering, enabling efficient data processing and insights into high-dimensional datasets.



 <br/><img src='/images/matrix-theory.png'>"
collection: portfolio
---
[Visit the GitHub Repository](https://github.com/gabrielkmbo/matrix-theory)

## Stack

Here are the technologies used in this project:

<p>
  <img src="/images/aws.png" alt="AWS" title="AWS" width="40" height="40" />
  <img src="/images/git.png" alt="Git" title="Git" width="40" height="40" />
  <img src="/images/python.png" alt="Python" title="Python" width="40" height="40" />
  <img src="/images/github.png" alt="Github" title="Github" width="40" height="40" />
  <img src="/images/pytorch.png" alt="Pytorch" title="Pytorch" width="40" height="40" />
  <img src="/images/numpy.png" alt="Numpy" title="Numpy" width="40" height="40" />
</p>

# Linear Algebra Tools for Data Science Applications

## About
This repository provides tools and scripts developed in **Python** to solve a variety of linear algebra and matrix theory problems. The goal is to deliver easy-to-use solutions for both academic and professional applications. These tools are particularly useful in data science workflows, supporting tasks such as:

- **Singular Value Decomposition (SVD)**
- **Principal Component Analysis (PCA)**
- **Linear Regression**
- **Least Squares Optimization**

## Features
- **Efficient Algorithms**:
  - Solve linear equations quickly and accurately.
- **Matrix Operations**:
  - Perform addition, multiplication, inversion, and other standard matrix computations.
- **Eigenvalue and Eigenvector Computation**:
  - Tools to compute and analyze eigenvalues and eigenvectors.
- **Complex Matrix Operations**:
  - Support for advanced operations involving complex matrices.

## Examples (Least Squares)
```python
import numpy as np

def least_squares(A, b, param):
        # b = b.reshape(-1, 1)

        n = A.shape[1]

        # make [A, sqrt(param)I] and [b, 0]
        A_prime = np.concatenate((A, np.sqrt(param) * np.eye(n)), axis=0)
        b_prime = np.concatenate((b, np.zeros(n)), axis=0)

        # use the least squares solver
        return np.linalg.lstsq(A_prime, b_prime, rcond=None)[0]

def main():
        A = np.array([[1, 2], [3, 4], [5, 6]])
        b = np.array([1, 2, 3])
        param = 0.1

        actual_solution = np.linalg.inv(A.T @ A + param * np.eye(A.shape[1])) @ A.T @ b
 
        print("Least Squares Solution: ")
        print(least_squares(A, b, param))
        print("Actual Solution: ")
        print(actual_solution)

if __name__ == "__main__":
        main()
```


## Applications
These tools can be integrated into various data science workflows, enhancing tasks such as dimensionality reduction, regression analysis, and optimization problems.

