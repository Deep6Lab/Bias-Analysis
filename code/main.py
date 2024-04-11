from data_preparation import DataLoader, DataPreprocessor, DataLabeler
from models import ModelManager
from analysis import ScaledBiasAnalysis, PromptBiasAnalysis
from visualizations import Phase1Visualization, Phase2Visualization
from utils import ResultsCompiler

def main():

    #loading the data
    data_loader = DataLoader(data_path="D:\work-colab\gh-repos\Deep6Lab\Bias-Analysis\data\IMDB_reviews.csv")
    raw_data = data_loader.load_data()

    #preprocessing the data
    raw_data['review'] = DataPreprocessor.remove_html_tags(raw_data['review'])
    raw_data['review'] = DataPreprocessor.preprocess(raw_data['review'])

    preprocessed_data = raw_data.copy()

    reviews = preprocessed_data['reviews'].tolist()

    model_manager = ModelManager(reviews, model_name = 'bert-base-uncased')
    bert_tokenizer, bert_model = model_manager.load_model()
    bert_embeddings = model_manager.get_embeddings(bert_tokenizer, bert_model)
    clusters = model_manager.kmeans_clusters(bert_embeddings)
    preprocessed_data['cluster'] = clusters

    #labeling the data
    openai_api_key = '#######'
    
    data_labeler = DataLabeler(preprocessed_data, openai_api_key, model='gpt-3.5-turbo')
    labeled_data = data_labeler.label_data()
    labeled_data.to_csv('D:\work-colab\gh-repos\Deep6Lab\Bias-Analysis\data\IMDB_labeled.csv', index=False)

    phase_1_plots = Phase1Visualization(labeled_data)
    phase_1_plots.generate_word_clouds() #based on these word clouds the target and attribute words are set
    phase_1_plots.generate_genre_distribution()
    phase_1_plots.generate_clusters_scatterplot()
    phase_1_plots.generate_embeddings_clusters()

    #bias analysis
    scaled_bias_analysis = ScaledBiasAnalysis(labeled_data)
    scaled_bias_results = scaled_bias_analysis.analyze_scaling_bias() #this generates 3 files one each for base,large,xlargemodels

    #prompt based analysis
    prompt_bias_analysis = PromptBiasAnalysis(labeled_data)
    prompt_bias_results = prompt_bias_analysis.analyze_bias_with_prompts()
    prompt_bias_results.to_csv('D:\work-colab\gh-repos\Deep6Lab\Bias-Analysis\results\model_bias_scores.csv', index=False)

    combined_results = pd.read_csv('D:\work-colab\gh-repos\Deep6Lab\Bias-Analysis\results\combined_results.csv') #manually pivoted in excel to create this file from the scaled_bias_analysis 3 files
    base = pd.read_csv('D:\work-colab\gh-repos\Deep6Lab\Bias-Analysis\results\model_bias_scores_base_models.csv')
    large = pd.read_csv('D:\work-colab\gh-repos\Deep6Lab\Bias-Analysis\results\model_bias_scores_large_models.csv')
    xlarge = pd.read_csv('D:\work-colab\gh-repos\Deep6Lab\Bias-Analysis\results\model_bias_scores_xlarge_models.csv')

    phase_2_plots = Phase2Visualization(combined_results, base, large, xlarge)
    phase_2_plots.score_base_models()
    phase_2_plots.score_large_models()
    phase_2_plots.score_xlarge_models()
    phase_2_plots.score_comparision_categories()

    results_compiler = ResultsCompiler(prompt_bias_results)
    results_compiler.combined_model_genre_comparision()
    results_compiler.seperate_model_genre_comparision()
    results_compiler.aggregate_genre_level_comparision()
    results.compiler.aggregate_model_level_comparision()


if _name_ == "_main_":
    main()