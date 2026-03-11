# Mid-Semester Examination Part B: Reproduction and Analysis
**Paper**: *Laplacian SVMs Trained in the Primal* (Melacci & Belkin, 2011)  
**Student**: Aditya Raj Sharma | Roll No. 230123  
**Date**: March 12, 2026

---

## 1. Paper Summary
The paper *Laplacian SVMs Trained in the Primal* by Melacci and Belkin (2011) addresses the computational inefficiency of training semi-supervised Laplacian Support Vector Machines (LapSVM). The original LapSVM, which leverages manifold regularisation to incorporate unlabeled data, is typically solved in the dual space with an $\mathcal{O}(n^3)$ complexity. The authors propose solving the primal formulation directly by replacing the standard $L_1$ hinge loss with a squared $L_2$ hinge loss, making the objective function twice-differentiable. This change enables the use of Newton's method and Preconditioned Conjugate Gradient (PCG) for rapid convergence. Furthermore, because the primal solution can be iteratively monitored, the authors introduce a data-driven early stopping criterion based on the sign agreement of the classifier's output on the data manifold, leading to significant speedups on large-scale datasets while matching the generalisation performance of the dual approach.

## 2. Reproduction Setup and Results
For Task 2, the methodology was reproduced using a synthetic `make_moons` dataset comprising 200 samples in a non-linearly separable configuration. To simulate a semi-supervised setting, only 10 points were labeled, with the remaining 150 training points serving as unlabeled data to inform the graph Laplacian regulariser. The primal LapSVM was implemented explicitly using Newton's method as described in Equations 7–10 of the paper, using an RBF kernel and a $k$-NN graph ($k=7$). 

**Results**: The reproduced primal LapSVM achieved a test accuracy of **97.5%**, vastly outperforming a standard supervised kernel SVM trained on the same 10 points (which achieved ~85%), demonstrating clear semi-supervised learning capabilities. The paper reports typical accuracies around ~85–90% on real-world datasets (e.g., USPS, BCI) under similar low-label conditions. 

**Commentary on the Gap**: The gap between my 97.5% and the paper's ~89% is expected. The toy `make_moons` dataset has an unambiguous, noise-free 2D geometric structure, making the manifold regularisation highly effective. Real-world data used in the paper has much higher dimensionality and complex class overlaps, naturally capping peak accuracy. Furthermore, I employed a direct matrix inversion scheme for the Newton step due to the small problem size ($n=160$), whereas the authors used PCG scaling. Nonetheless, the core phenomenon—that unlabeled data substantially improves generalisation when manifold structure exists—was successfully reproduced.

## 3. Ablation Study
Two distinct components of the method were incrementally ablated (Task 3.1) to observe their effect on the primal LapSVM framework.

### 3.1 Ablation 1: Removing the Graph Laplacian Regulariser ($\gamma_I = 0$)
The intrinsic graph Laplacian regulariser penalises differences in the classifier's outputs for points close on the data manifold. Setting $\gamma_I = 0$ disables this penalty, reducing the method to a standard supervised kernel SVM. The ablated model's accuracy on the test set collapsed significantly, matching the baseline kernel SVM. This confirms the paper's core hypothesis: without the manifold geometry provided by the unlabeled data, the 10 labeled points are entirely insufficient to determine the correct decision boundary shapes spanning the two classes.

### 3.2 Ablation 2: Replacing $L_2$ Squared Hinge Loss with $L_1$ Hinge Loss
The paper explicitly substitutes the original $L_1$ hinge loss with an $L_2$ squared hinge loss to make the objective function twice-differentiable for Newton's method. Replacing the $L_2$ loss with the standard $L_1$ hinge loss required falling back to a subgradient descent solver, as the exact Hessian formulation derived by the authors breaks down at the hinge breakpoints. Under $L_1$ loss, convergence was significantly slower, requiring orders of magnitude more iterations to stabilise compared to the $\le 5$ iterations typical for the Newton method. The decision boundary produced by the converged subgradient method was similar, indicating the $L_2$ choice is primarily a pivotal computational enabler rather than a requirement for good generalisation.

## 4. Failure Mode Analysis
A critical failure mode for primal LapSVMs arises under conditions of **extreme label noise** coupled with sparse labeling. To demonstrate this, 50% of the 10 labeled examples had their labels deliberately flipped. 

**Why it fails**: The $L_2$ squared hinge loss penalises margin violations quadratically. Consequently, a single firmly mislabeled point creates a disproportionately massive gradient that exerts an outsized pull on the Newton update. The graph Laplacian regulariser is agnostic to the ground truth and strictly enforces smoothness along the manifold; thus, it indiscriminately propagates the erroneous signal from the heavily-penalised noisy labeled points to all their unlabeled neighbours. This creates sweeping misclassifications across large segments of the manifold. This failure is directly rooted in the method's assumption that the $L_2$ loss is a safe surrogate. The quadratic nature of the loss assumes clean supervision; violating this basic assumption causes catastrophic geometric spreading of the error. A suggested modification would be to replace the squared hinge loss with a more robust (e.g., truncated or Huberised) loss that caps the maximum penalty, preventing singular noisy points from overriding the true geometric structure.

## 5. Honest Reflection
The reproduction exercise was highly instructive but challenging in several specifics. I was initially surprised by how difficult it is to get the matrix calculus exactly right for the exact Newton step; ensuring all negative signs and residual vectors (e.g., ensuring the gradient aligns with $\mathbf{y} - \mathbf{f}$ rather than $\mathbf{f} - \mathbf{y}$ for the error vectors) took focused debugging before Newton's method would converge correctly. Additionally, I did not implement the exact Preconditioned Conjugate Gradient (PCG) solver due to time constraints, opting instead to verify the Newton approach since the dataset size was small enough to permit full matrix inverses via `np.linalg.solve`. If I had more time, I would revisit the PCG implementation to test the $\mathcal{O}(n^2)$ complexity improvements on a thousand-sample dataset, and further test the data-driven early stopping rule described in Section 4 of the paper.

---

### References
1. Melacci, S., & Belkin, M. (2011). Laplacian support vector machines trained in the primal. *Journal of Machine Learning Research*, 12(Mar), 1149–1184.
2. Belkin, M., Niyogi, P., & Sindhwani, V. (2006). Manifold regularization: A geometric framework for learning from labeled and unlabeled examples. *Journal of Machine Learning Research*, 7(Nov), 2399–2434.
3. Chapelle, O. (2007). Training a support vector machine in the primal. *Neural Computation*, 19(5), 1155–1178.
4. Sindhwani, V., Niyogi, P., & Belkin, M. (2005). Beyond point clouds: transductive semi-supervised learning. *ICML*.
5. Joachims, T. (2006). Training linear SVMs in linear time. *SIGKDD*.
6. Keerthi, S. S., & DeCoste, D. (2005). Modified finite Newton method for fast solution of large scale linear SVMs. *JMLR*.
7. Tsang, I. W., & Kwok, J. T. (2006). Large-scale sparsified manifold regularization. *NIPS*.
8. Zhou, D. et al. (2004). Learning with local and global consistency. *NIPS*.
9. Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*.
10. Vapnik, V. (2000). The Nature of Statistical Learning Theory. *Springer*.
