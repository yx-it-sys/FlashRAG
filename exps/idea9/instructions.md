# Modify the Budget allocations
Core Logic: From "Subjective Importance" to "Objective Discriminative Power"
The core idea of Entropy-QV is: if a voter's evaluations of all candidates are roughly the same, then that voter's vote is noise and their budget should be reduced.

## Calculation of Entropy of Preference Distribution
For each Clue $c_i$, its preference score vector for the candidate document set $S$ is $S_i = [S_{i,1}, S_{i,2}, \dots, S_{i,k}]$. We first normalize it into a probability distribution $P_i$:
$$P_{i,j} = \frac{S_{i,j}}{\sum_{m=1}^{k} S_{i,m}}$$
Subsequently, calculate the information entropy (Shannon Entropy) of this clue:
$$H(c_i) = -\sum_{j=1}^{k} P_{i,j} \log(P_{i,j})$$
- High Entropy: It means that the distribution of $S_{i,j}$ is flat, and this clue cannot distinguish candidate documents, with weak discriminative power. 
- Low Entropy: It means that the distribution of $S_{i,j}$ is sharp (concentrated on a few documents), and this clue has a strong directivity.

## Discriminative Weight
We define a discriminative weight $w_i^{dist}$ to modify the original importance score $r_i$:
$$w_i^{dist} = 1 - \frac{H(c_i)}{\log(k)}$$
where $\log(k)$ is the maximum possible value of entropy (uniform distribution). The closer $w_i^{dist}$ is to 1, the stronger the obstacle-avoiding ability of the clue.

Under the Entropy-QV framework, the revised budget allocation formula, formula (2), is reconstructed as:
$$\hat{I}_i = \frac{r_i \cdot w_i^{dist}}{\sum_{m=1}^{|C|} (r_m \cdot w_m^{dist})} I_{total}$$

In this way, even if the VLM considers a certain clue $c_i$ to be semantically important (with a high $r_i$), its actual voting influence $\hat{I}_i$ will be suppressed if it cannot stand out in the current candidate pool (with a low $w_i^{dist}$).