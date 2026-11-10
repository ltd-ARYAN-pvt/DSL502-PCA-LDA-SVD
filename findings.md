## Q1
Assignment asks which component range captures more visual and structural details.

Our experimental results support the following conclusions:
- PC1–PC3 captures major facial structure and broad intensity variations.
- PC5–PC9 captures lower-variance patterns but cannot independently preserve the dominant facial structure.
- PC1–PC9 provides the best overall numerical reconstruction.
- Nolights has a larger proportion of variance concentrated in PC1, whereas Lights distributes its variance somewhat differently.
- PCA captures illumination-related variations as well as facial variations; its principal components are not necessarily identity-specific.
---

## Q2
PCA preserves directions of maximum overall variance, which may include variations caused by illumination and facial identity. LDA, in contrast, uses class labels to maximize between-class separation relative to within-class variation. Consequently, the LDA projections show substantially more compact and distinct person-wise clusters than the PCA projections under both lighting conditions.
The Fisher separation ratio provides quantitative evidence of the superiority of LDA over PCA for person-wise discrimination. Under the lights condition, the ratio increased from 0.5142 (PCA) to 622.1718 (LDA). Under the nolights condition, it increased from 0.2563 to 270.6606. These results indicate that LDA produces significantly more discriminative two-dimensional representations by maximizing between-class scatter relative to within-class scatter.
The LDA model was evaluated using ten different combinations of three subjects selected from the five available subjects. Under the lights condition, the separation ratios ranged from 505.00 to 2034.61, whereas under the nolights condition, they ranged from 103.66 to 836.74. The combination of Person9, Person19, and Person46 achieved the highest separation ratio under the lights condition (2034.61). Under the nolights condition, Person2, Person9, and Person46 achieved the highest ratio (836.74). These results demonstrate that LDA performance depends on the selected class combination and the illumination condition.
---

## Q3
The MSE analysis shows that retaining only the top five singular values produces the lowest reconstruction error for all three images. The average MSE increases from 247.71 for five components to 298.76 for twenty components and 478.67 for fifty components. This indicates that retaining more singular components also preserves a greater amount of noise. However, the top-five reconstruction exhibits stronger smoothing and loss of fine facial details. Thus, the lowest MSE does not necessarily correspond to the best perceptual reconstruction.
The experiment demonstrates that truncated SVD can reduce image noise by retaining only a selected number of dominant singular components. For Gaussian noise, the average MSE decreased from 542.60 in the noisy images to 247.71 using five components. For Salt-and-Pepper noise, it decreased from 1105.96 to 324.00. The results also show that the optimal number of retained components depends on noise intensity. At a Gaussian noise standard deviation of 10, twenty components produced the lowest average MSE (60.17), whereas five components performed best at standard deviations of 25 and 50. This illustrates the trade-off between noise suppression and preservation of facial details.

## Q4: Movie Recommendation using Singular Value Decomposition

The MovieLens 100K dataset containing 100,000 ratings from 943 users across 1,682 movies was used to develop a collaborative filtering recommendation system.

A user–movie rating matrix of dimensions 943 × 1682 was constructed. The ratings were divided into 80% training data (80,000 ratings) and 20% testing data (20,000 ratings). User-wise mean-centering was applied to the training matrix before performing manual Singular Value Decomposition using eigenvalue decomposition of the Gram matrix.

The model was evaluated using five different latent dimensions: k = 5, 10, 20, 50 and 100. The best performance was obtained at k = 10, with an MAE of 0.780559 and RMSE of 0.987493.

The cumulative singular-value energy analysis indicated that 321 components were required to retain approximately 90% of the total energy. However, the lowest prediction error was achieved using only 10 components. This demonstrates that maximum energy preservation does not necessarily correspond to optimal recommendation accuracy.

Finally, the model generated personalized Top-10 movie recommendations for five selected users by ranking predicted ratings and excluding movies already rated in the training dataset.