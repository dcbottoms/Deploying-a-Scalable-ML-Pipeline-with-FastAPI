# Model Card

For additional information, see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This model is a supervised binary classification model trained to predict whether an individual earns more than $50,000 per year based on demographic and employment-related census data.

The model uses a Random Forest classifier from scikit-learn. Categorical features are processed using one-hot encoding, and the target label is binarized before training. The trained model, encoder, and label binarizer are saved as serialized `.pkl` files and reused during API inference.

The model was developed as part of the Udacity Machine Learning DevOps project, "Deploying a Machine Learning Model with FastAPI."

## Intended Use

The intended use of this model is to demonstrate a complete machine learning deployment workflow. The project includes data preprocessing, model training, model evaluation, unit testing, model serialization, FastAPI endpoint creation, and local API interaction.

This model is intended for educational purposes only. It should not be used to make real employment, lending, compensation, eligibility, or other high-impact decisions about individuals.

## Training Data

The model was trained using the Census Income dataset provided with the project starter repository. The dataset contains demographic and employment-related attributes including age, workclass, education, marital status, occupation, relationship, race, sex, capital gain, capital loss, hours worked per week, native country, and salary.

The target variable is `salary`, which contains two possible values: `<=50K` and `>50K`.

The categorical features used by the model are `workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, and `native-country`.

The numeric features used by the model are `age`, `fnlgt`, `education-num`, `capital-gain`, `capital-loss`, and `hours-per-week`.

The dataset was split into training and test sets using an 80/20 split.

## Evaluation Data

The evaluation data is the test portion of the Census Income dataset created during the train/test split. The test set was not used to fit the model, encoder, or label binarizer.

The same preprocessing steps were applied to the test data using the encoder and label binarizer fitted on the training data. This ensured that training and evaluation used consistent feature transformations.

## Metrics

The model was evaluated using precision, recall, and F1 score.

Precision measures how many of the model's positive predictions were correct. Recall measures how many actual positive cases the model correctly identified. F1 score provides a combined measure of precision and recall.

The overall model performance on the test set was:

- Precision: 0.7419
- Recall: 0.6384
- F1 Score: 0.6863

The model was also evaluated on categorical data slices. Slice performance was calculated for every unique value of each categorical feature, and the results were saved in `slice_output.txt`.

Examples of slice performance include:

- For `workclass = Federal-gov`, the model achieved precision of 0.7971, recall of 0.7857, and F1 score of 0.7914.
- For `workclass = Local-gov`, the model achieved precision of 0.7576, recall of 0.6818, and F1 score of 0.7177.
- For `workclass = Private`, the model achieved precision of 0.7376, recall of 0.6404, and F1 score of 0.6856.

## Ethical Considerations

This dataset includes sensitive and demographic attributes such as race, sex, native country, marital status, and relationship status. A model trained on these features may learn or amplify historical patterns of inequality present in the data.

Because of this, the model should not be used for real-world decision-making about individuals. It is especially inappropriate for use in employment, compensation, credit, lending, housing, insurance, education, or other high-impact contexts.

The model is best understood as a technical demonstration of machine learning engineering practices, not as a fair or validated decision system.

## Caveats and Recommendations

This model has several limitations. The data may contain historical bias, and the model may produce different performance levels across demographic slices. The model has only been evaluated using precision, recall, F1 score, and categorical slice metrics.

Before any real-world use, additional fairness analysis, bias testing, explainability analysis, and validation on more recent and representative data would be necessary.

The model should be retrained and re-evaluated if the underlying data distribution changes. Additional monitoring would also be needed in a production environment.
