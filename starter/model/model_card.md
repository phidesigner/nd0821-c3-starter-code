# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
The model predicts whether a person's income exceeds $50K/year using the Census Income dataset. The dataset contains information on demographic and employment-related attributes.

## Intended Use
The model is intended for educational and research purposes to understand income prediction using demographic and employment features. It is not suitable for making financial decisions about individuals.

## Training Data
The Census Income dataset from the UCI Machine Learning Repository was used for training. It contains 32,561 instances with 14 attributes.

## Evaluation Data
The model was evaluated on a holdout test dataset, representing 20% of the original data. It contains 6,452 instances.

## Metrics
The model was evaluated using precision, recall, and F1 score. Overall performance metrics on the test dataset:
- Precision: 0.738
- Recall: 0.638
- F1 Score: 0.684

Metrics for slices based on different categorical features shows some variability e.g. for the `workclass`:
- Private: F1 Score = 0.686
- Federal-gov: F1 Score = 0.791
Complete slice metrics are available in the associated logs file `model/logs/slice_metrics.log`.

### Summary table of slice metrics:

| Feature | Values | Total Samples | Avg. Precision | Avg. Recall | Avg. F1 |
|---------|--------|---------------|----------------|-------------|---------|
| education | Some-college, HS-grad, Bachelors, Masters, Assoc-acdm, 7th-8th, 11th, Assoc-voc, Prof-school, 9th, 5th-6th, 10th, Doctorate, 12th, 1st-4th, Preschool | 6513 | 0.772 | 0.575 | 0.630 |
| marital-status | Divorced, Married-civ-spouse, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse | 6513 | 0.904 | 0.331 | 0.434 |
| native-country | United-States, Mexico, Puerto-Rico, El-Salvador, ?, Columbia, Poland, Cambodia, Germany, Philippines, Canada, Ecuador, Italy, Cuba, Iran, Jamaica, Haiti, South, Taiwan, Dominican-Republic, Ireland, Japan, Scotland, India, Trinadad&Tobago, Hungary, England, Laos, Hong, Greece, Portugal, Guatemala, China, France, Nicaragua, Vietnam, Thailand, Peru, Honduras, Yugoslavia | 6513 | 0.803 | 0.772 | 0.749 |
| occupation | Adm-clerical, Exec-managerial, Machine-op-inspct, Craft-repair, Prof-specialty, Sales, Handlers-cleaners, Other-service, Protective-serv, Priv-house-serv, Transport-moving, ?, Farming-fishing, Tech-support, Armed-Forces | 6513 | 0.736 | 0.563 | 0.618 |
| race | White, Black, Other, Asian-Pac-Islander, Amer-Indian-Eskimo | 6513 | 0.776 | 0.626 | 0.691 |
| relationship | Not-in-family, Wife, Husband, Unmarried, Own-child, Other-relative | 6513 | 0.862 | 0.434 | 0.536 |
| sex | Female, Male | 6513 | 0.734 | 0.587 | 0.651 |
| workclass | Private, State-gov, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, ?, Without-pay | 6513 | 0.772 | 0.679 | 0.718 |


## Ethical Considerations
Potential biases may exist in the data due to historical and societal inequities e.g., predictions may vary across demographic groups such as race, gender, or workclass. This should be taken into account when interpreting model results.

## Caveats and Recommendations
The model is limited by the quality and representativeness of the training data. It may not generalize well to other datasets or populations. Users should carefully assess the model's suitability for their specific use case and be aware of potential ethical concerns.
