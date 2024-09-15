# Ellipsoid Fit Implementation

This script aims to implement the ellipsoid fit method proposed in [1] to develop a method for magnetic compass calibration. As the algebraic analysis presented in section (IV) of the paper is not enough for the direct implementation, we will take the algebraic analysis further for the straightforward Python implementation.

[1] Liu, Y. X., Li, X. S., Zhang, X. J., & Feng, Y. B. (2014). Novel calibration algorithm for a three-axis strapdown magnetometer. Sensors, 14(5), 8485-8504.

## Magnetic Calibration method

The measurement of a magnetic compass can be modeled as:

$$ h_m^b = Mh^b + b + n $$

Where $b$ is a constant offset that shifts the output of the sensors, and $M$ accounts for the the sensitivity of the individual axes of the sensor, the non-orthogonality and misalignment of the axes, and the sum of soft-iron errors fixed to the body frame, which serves to scale the sensor's output.

When the magnetic compass remains stationary and only changes direction, the magnitude of the true magnetic field $||h^b||$ remains constant, and the locus of the true magnetic field measured $h^b$ is a sphere. Meanwhile, the locus of the disturbed magnetic field measured $h_m^b$ is an ellipsoid, and it can be expressed as follows:

$$ ||h^b||^2 = (h_m^b)^TAh_m^b - 2b^TAh_m^b + b^TAb + \tilde{n} $$

Where $A = G^TG$, $G = M^{-1}$, and $\tilde{n} = 2(h_m^b-b)^TG^TGn + n^TG^TGn$. We can see that this is the expression of an ellipsoid in terms of $h_m^b$. In other words, the measurements $(h_m^b)$ with errors are constrained to lie on an ellipsoid. Thus, the calibration of the magnetic compass is to seek ellipsoid-fitting methods to solve the coefficients of $G$ and $b$.

Since an ellipsoid is a kind of concoid, the ellipsoid equation can be expressed as a general equation of a concoid in the 3-D space as follows:

$$ F(a, h_m^b) = a(h_{mx}^b)^2 + b(h_{mx}^bh_{my}^b) + c (h_{my}^b)^2 + d(h_{mx}^bh_{mz}^b) + e(h_{my}^bh_{mz}^b) + j(h_{mz}^b)^2 + p(h_{mx}^b) + q(h_{my}^b) + r(h_{mz}^b) + s = 0$$

Where $a = \begin{bmatrix} a && b && c && d && e && j && p && q && r && s\end{bmatrix}^T$. Moreover, the problem of fitting an ellipsoid into N data points $h_m^b$ can be solved by minimizing the sum of squares of the algebraic distance:

$$ min \; \sum_{i}^nF_i(a, h_m^b)^2$$

We define the design matrix (we will use the notation $h_j^i$ to denote the i-th sample of $h_{mj}^b$), where the i-th row is:

$$ S_i = \begin{bmatrix} (h_{x}^i)^2 && h_{x}^ih_{y}^i && (h_{y}^i)^2 && h_{x}^ih_{z}^i && h_{y}^ih_{z}^i && (h_{z}^i)^2 && h_{x}^i && h_{y}^i && h_{z}^i && 1 \end{bmatrix} $$

An additional property of the design matrix is that (S^TS) is symmetric:

$$ (S^TS)^T = (S)^T(S^T)^T = S^TS \; \rightarrow \; (S^TS)^T = S^TS $$

Now, the minimization problem can be presented as:

$$ min \; ||Sa||^2 \; \rightarrow \; min \; (Sa)^T(Sa) \; \rightarrow \; min \; a^TS^TSa$$

In order to avoid the trivial solution $a = \mathbb{O}_{10}$, and recognizing that any multiple of a solution $a$ represents the same concoid, the parameter vector $a$ is constrained in someway. In order that the surface is fitted to be an ellipsoid in 3D, the parameters $a$ must insure the matrix $A$ to be either positive or negative definite, the equivalent constrained condition is:

$$4ac - b^2 > 0$$

The imposition of this inequality constraint is difficult in general; in this case, we have the freedom to arbitrarily scale the parameters, so we may simply incorporate the scaling into the constraint and impose the equality constraint $$4ac - b^2 = 1$$, which can be expressed in the matrix form of $a^TCa = 1$, as:

$$ C = \begin{bmatrix} C_1 && C_2 \\ C_3 && C_4 \end{bmatrix},\; \text{where:} \; C_1 = \begin{bmatrix} 0 && 0 && 2 \\ 0 && -1 && 0 \\ 2 && 0 && 0 \end{bmatrix}, \; C_2 = \mathbb{O}_{3 \times 7}, \; C_3 = \mathbb{O}_{7 \times 3}, \; C_4 = \mathbb{O}_{7 \times 7}$$

Notice that as $C$ is a block matrix, $C_1 = C_1^T$, and all the other blocks are zeros:

$$ C^T = \begin{bmatrix} C_1^T && C_3^T \\ C_2^T && C_4^T \end{bmatrix} \; \rightarrow \; C^T = \begin{bmatrix} C_1 && C_2 \\ C_3 && C_4 \end{bmatrix} = C$$

With this additional constraint, the optimization problem is:

$$ min \; a^TS^TSa \quad s.t. \quad a^TCa - 1= 0$$

Using the Lagrange method:

$$ \mathcal{L}(a) = a^TS^TSa - \lambda(a^TCa - 1= 0) $$

Differentiating with respect to $a$ we find:

$$ \frac{\partial \mathcal{L}}{\partial a} = 0 \; \rightarrow \; a^T(S^TS + SS^T) - \lambda(a^T(C + C^T)) = 0 \; \rightarrow \; 2a^TS^TS - 2\lambda a^TC = 0 \; \rightarrow \; S^TSa = \lambda Ca$$

To solve this system, we can rewrite the matrix $S^TS$ as a block matrix:

$$ S^TS = \begin{bmatrix} X_{11} && X_{12} \\ X_{21} && X_{22} \end{bmatrix}, \text{ where: } X_{11} \text{ is } 3 \times 3, \; X_{12} \text{ is } 3 \times 7, \; X_{21} \text{ is } 7 \times 3, \; X_{22} \text{ is } 7 \times 7$$

Furthermore, Using the definition that $S^TS$ is symmetric, then: $X_{21} = X_{12}^T$:

$$ S^TS = \begin{bmatrix} X_{11} && X_{12} \\ X_{12}^T && X_{22} \end{bmatrix}$$

We can also define: $a_1 = \begin{bmatrix} a && b && c\end{bmatrix}^T$ and $a_2 = \begin{bmatrix} d && e && j && p && q && r && s\end{bmatrix}^T$, such that: $a = \begin{bmatrix} a_1 && a_2\end{bmatrix}^T$. Now, we can rewrite the system as:

$$ \begin{bmatrix} X_{11} && X_{12} \\ X_{12}^T && X_{22} \end{bmatrix}\begin{bmatrix} a_1 \\ a_2\end{bmatrix} = \lambda \begin{bmatrix} C_1 && C_2 \\ C_3 && C_4 \end{bmatrix}\begin{bmatrix} a_1 \\ a_2\end{bmatrix} $$

Considering that $C_2, C_3, C_4$ are zero matrices, the first equation that we can get from this is:

$$ X_{11}a_1 + X_{12}a_2 = \lambda C_1a_1 $$

The second equation is:

$$ X_{12}^Ta_1 + X_{22}a_2 = 0 $$

If the data is not coplanar, the $X_{22}$ will be non-singular:

$$ \rightarrow a_2 = -X_{22}^{-1}X_{12}^Ta_1 $$

Replacing this term in the first equation, and considering that $C_1$ is non-singular as $det(C_1) = 4$:

$$(X_{11} - \lambda C_1)a_1 + X_{12}(-X_{22}^{-1}X_{12}^Ta_1) = 0 \; \rightarrow \; C_1^{-1}(X_{11} - X_{12}X_{22}^{-1}X_{12}^T)a_1 = \lambda a_1 $$

It is proven that **exactly one eigenvalue** of the last system is positive. Let $u_1$ be the eigenvector associated with the only positive eigenvalue of the general system, then we can get the full solution: $u = \begin{bmatrix} u_1 && u_2\end{bmatrix}^T$. In the case that the matrix $(X_{11} - X_{12}X_{22}^{-1}X_{12}^T)$ is singular, the corresponding $u_1$ can be replaced with the eigenvector associated with the largest eigenvalue.

With the previously defined equations, we are able to determine the values of: $a = \begin{bmatrix} a && b && c && d && e && j && p && q && r && s\end{bmatrix}^T$, and with those values we need to recover the soft-iron and hard-iron, $M$ and $b$, respectively. By definition:

### Soft-Iron

$$ A = \begin{bmatrix} a && b/2 && d/2 \\ b/2 && c && e/2 \\ d/2 && e/2 && j \end{bmatrix} $$

And we also defined that: $G^TG = A$, with $G = M^{-1}$. However, there are uncountable matrices $G$ which can satisfy $G^TG = A$. Thus, we select the x-axis of the sensors as the x-axis of the magnetic compass; thus we can obtain the unique $G$ by singular value decomposition.

$$ G = U_G\Sigma_G V_G^T \; \rightarrow G^TG = (V_G\Sigma_G^TU_G^T)(U_G\Sigma_G V_G^T) \; \leftrightarrow \; G^TG = V_G\Sigma_G^T\Sigma_G V_G^T $$

As A is a symmetric matrix, then: $A^T = A \; \rightarrow \; G^TG = GG^T$.

$$ G = U_G\Sigma_G V_G^T \; \rightarrow GG^T = (U_G\Sigma_G V_G^T)(V_G\Sigma_G^T U_G^T) \; \leftrightarrow \; GG^T = U_G\Sigma_G \Sigma_G^T U_G^T $$

If we do the singular value decomposition of A:

$$ V_G\Sigma_G^T\Sigma_G V_G^T = U_A \Sigma_A V_A^T \qquad \qquad U_G\Sigma_G\Sigma_G^T U_G^T = U_A \Sigma_A V_A^T $$

Then, as $A$ is a symmetric matrix, then: $U = V$, therefore:

$$ U_G = U_A \qquad \qquad V_G = V_A \qquad \qquad \Sigma_G = \Sigma_A^{1/2} $$

Then:

$$ G = U_G \Sigma_G V_G^T \quad \rightarrow \quad G = U_A \Sigma_A^{1/2} V_A^T \quad \rightarrow \quad M = (U_A \Sigma_A^{1/2} V_A^T)^{-1} $$

### Hard-Iron

From the magnetic field magnitude we found that:

$$ ||h^b||^2 = (h_m^b)^TAh_m^b - 2b^TAh_m^b + b^TAb + \tilde{n} $$

The first term of that equation is related to the quadratic terms, while the second term is related to the linear terms:

$$ - 2b^TAh_m^b = \begin{matrix}h_{mx} \left(- 2 a b_{x} - b b_{y} - b_{z} e\right) + h_{my} \left(- b b_{x} - 2 b_{y} c - b_{z} d\right) + h_{mz} \left(- b_{x} e - b_{y} d - 2 b_{z} j\right)\end{matrix} $$

From the general equation of a concoid in the 3-D space, the linear terms are:

$$ p(h_{mx}^b) + q(h_{my}^b) + r(h_{mz}^b) $$

Therefore, we have the system:

$$ \begin{bmatrix} 2a && b && d \\ b && 2c && e \\ d && e && 2j \end{bmatrix} \begin{bmatrix} b_x \\ b_y \\ b_z \end{bmatrix} = \begin{bmatrix} p \\ q \\ r \end{bmatrix} \; \rightarrow \;  \begin{bmatrix} b_x \\ b_y \\ b_z \end{bmatrix} = \begin{bmatrix} 2a && b && d \\ b && 2c && e \\ d && e && 2j \end{bmatrix} ^{-1} \begin{bmatrix} p \\ q \\ r \end{bmatrix} \; \leftrightarrow \;  \begin{bmatrix} b_x \\ b_y \\ b_z \end{bmatrix} = (2A)^{-1} \begin{bmatrix} p \\ q \\ r \end{bmatrix}$$

::: magyc.benchmark_methods.ellipsoidfit
