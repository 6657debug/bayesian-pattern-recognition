# AI prompts used for the assignment

The assignment asks for an AI tool to handle algorithm search, programming, and report writing. These are the representative prompts used to guide that work.

1. **Find and verify the algorithm**

   > Find a recent classification method for computer vision or pattern recognition that is explicitly based on probability or Bayesian theory. Prefer a 2023-or-later peer-reviewed primary source and an author-maintained implementation. Explain why the method fits this assignment and state what part can be reproduced on a small, offline dataset.

   The source search used queries including “Variational Bayesian Last Layers ICLR 2024 classification”, “VBLL official implementation variational Bayesian last layer code”, and “Variational Bayesian Last Layers ELBO formula classification”. The ICLR paper and its theorem were checked before implementation.

2. **Plan the experiment**

   > Propose a reproducible experiment for the discriminative Variational Bayesian Last Layers classifier using a built-in handwritten-image dataset. Specify the data split, fixed random seed, deterministic baseline, validation-only checkpoint selection, uncertainty metrics, and limitations. Do not use test data to tune the method.

3. **Generate the implementation**

   > Generate a complete Python script that trains a standard MLP, freezes its hidden features, and post-trains a discriminative VBLL head with independent diagonal Gaussian weight rows by maximizing the sampling-free Jensen classification bound plus the analytic Gaussian KL term divided by the number of training examples. Initialize the variational mean from the trained softmax head, report validation-selected checkpoint results, and estimate test predictive probabilities by sampling from q(W). Include deterministic settings and save machine-readable results.

4. **Interpret results**

   > Read the actual experiment output and compare accuracy, NLL, Brier score, 15-bin ECE, and entropy-based error-detection AUROC. Explain what improved and what became worse without claiming that Bayesian inference must outperform the deterministic baseline.

5. **Write and check the report**

   > Write an English report that answers all six items in the assignment, distinguishes the cited paper from this smaller reproduction, documents the AI prompts and generated implementation, states the observed metrics and limitations, and links to the source-code repository.

The final script and report were produced with AI assistance. The code was executed to obtain the reported results; numerical values in the report are taken from `experiment_results.json`.
