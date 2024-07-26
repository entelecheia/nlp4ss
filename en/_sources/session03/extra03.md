# Session 3 Extra Notes

## 1. Cost Management

When using commercial LLM APIs like OpenAI's GPT models, it's crucial to consider the cost implications:

- API calls are typically charged per token processed
- Costs can quickly accumulate when processing large datasets
- Start with smaller, cheaper models (e.g., GPT-3.5 instead of GPT-4) for initial testing
- Use a small sample of your data (e.g., 100 examples) to develop and refine your approach before scaling up

## 2. Technical Setup

Ensure your environment is properly configured:

- When installing new packages, use the "Restart Session" option in your notebook environment to ensure they are correctly loaded
- Be aware of version compatibility issues, especially with libraries like NumPy
- Consider using virtual environments to manage dependencies

## 3. Data Handling

Efficient data handling is key when working with large datasets:

- Start with a small subset of your data for development and testing
- Save intermediate results to avoid re-running expensive operations
- Consider preprocessing steps that can reduce the amount of text sent to the LLM

## 4. Prompt Engineering

Effective prompt design is crucial for getting desired results from LLMs:

- Be explicit and specific in your instructions
- Include constraints (e.g., "Only respond with the sentiment label, without any additional explanation")
- Use examples to demonstrate the desired output format (few-shot learning)
- Iterate on your prompts to improve consistency and accuracy

## 5. Output Validation

LLM outputs need careful validation:

- Manually review a sample of outputs to check for consistency and accuracy
- Implement automated checks for expected output formats
- Be prepared to refine your approach based on observed issues

## 6. Alternatives to Commercial APIs

Consider alternatives to commercial LLM APIs:

- Open-source models can be run locally, though they may require more technical setup
- Some models can run on consumer-grade hardware, offering a cost-effective solution for smaller projects
- Research-focused models or APIs may offer discounts for academic use

## 7. Reproducibility

Ensure your research is reproducible:

- Document your exact prompts and any refinements made
- Record the specific model versions used
- Save raw outputs along with your processed results

## 8. Hybrid Approaches

Consider combining LLM-based methods with traditional NLP techniques:

- Use LLMs for complex tasks or initial data exploration
- Validate or refine LLM outputs using rule-based systems or smaller, task-specific models
- Leverage LLMs to generate training data for traditional supervised learning models
