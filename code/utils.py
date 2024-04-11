import numpy as np
import matplotlib.pyplot as plt

class ResultsComplier:
    def __init__(self, data):
        self.data = data
    
    @staticmethod
    def max_abs_with_sign(series):
        max_abs_value = series.abs().max()
        max_value = series[series.abs() == max_abs_value].iloc[0]
        return max_value

    @staticmethod
    def min_abs_with_sign(series):
        min_abs_value = series.abs().min()
        min_value = series[series.abs() == min_abs_value].iloc[0]
        return min_value

    def combined_model_genre_comparision(self):
        corrected_extreme_scores = self.data.groupby(['Model', 'Genre']).agg({'Score': max_abs_with_sign, 'PromptedScore': min_abs_with_sign}).reset_index()
        corrected_extreme_scores['PercentageChange'] = ((corrected_extreme_scores['Score'] - corrected_extreme_scores['PromptedScore']) / corrected_extreme_scores['Score']) * 100
        fig, ax = plt.subplots(figsize=(14, 10))
        n_combinations = len(corrected_extreme_scores)
        index = np.arange(n_combinations)
        bar_width = 0.35
        opacity = 0.8
        bar1 = ax.bar(index, corrected_extreme_scores['Score'], bar_width, alpha=opacity, label='Unprompted Score')
        bar2 = ax.bar(index + bar_width, corrected_extreme_scores['PromptedScore'], bar_width, alpha=opacity, label='Prompted Score')
        for idx, row in corrected_extreme_scores.iterrows():
            height = max(row['Score'], row['PromptedScore'])
            ax.text(idx + bar_width/2, height, f"{row['PercentageChange']:.2f}%", ha='center', va='bottom')
        ax.set_xlabel('Model-Genre Combination')
        ax.set_ylabel('Scores')
        ax.set_title('Combined Model-Genre Level Effect Scores Comparison before and after Prompting')
        ax.set_xticks(index + bar_width / 2)
        ax.set_xticklabels(corrected_extreme_scores['Model'] + "-" + corrected_extreme_scores['Genre'], rotation=45, ha="right")
        ax.legend()
        fig.tight_layout()
        plt.savefig('D:\work-colab\gh-repos\Deep6Lab\Bias-Analysis\results\combined_modelgenre_scores.png')
        plt.show()

    def seperate_model_genre_comparision(self):
        corrected_extreme_scores = self.data.groupby(['Model', 'Genre']).agg({'Score': max_abs_with_sign, 'PromptedScore': min_abs_with_sign}).reset_index()
        fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(20, 5))
        models = corrected_extreme_scores["Model"].unique()
        for i, model in enumerate(models):
            ax = axs.flat[i]
            data = corrected_extreme_scores[corrected_extreme_scores["Model"] == model]
            x = list(range(len(data["Genre"])))
            y1 = data["Score"]
            y2 = data["PromptedScore"]
            width = 0.3
            ax.bar(x, y1, width, color="lightcoral", label="Before Prompt")
            ax.bar([i + width for i in x], y2, width, color="lightgreen", label="After Prompt")
            ax.set_ylabel("Bias Score")
            ax.set_title(model)
            ax.set_xticks(x)
            ax.set_xticklabels(data["Genre"].unique())
            ax.legend()
            for j, (before, after) in enumerate(zip(y1, y2)):
                ax.text(j, before, f'{before:.2f}', ha='center', va='bottom')
                ax.text(j + width, after, f'{after:.2f}', ha='center', va='bottom')
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.suptitle('Seperate Model Genre Level Effect Scores Comparison before and after Prompting')
        plt.savefig('D:\work-colab\gh-repos\Deep6Lab\Bias-Analysis\results\seperate_modelgenre_scores.png', dpi=300)
        plt.show()

    def aggregate_genre_level_comparision(self):
        corrected_extreme_scores = self.data.groupby(['Model', 'Genre']).agg({'Score': max_abs_with_sign, 'PromptedScore': min_abs_with_sign}).reset_index()
        new_df = corrected_extreme_scores.groupby('Genre').agg({'Score': 'mean', 'PromptedScore': 'mean'}).reset_index()
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
        new_df['PercentageChange'] = ((new_df['Score'] - new_df['PromptedScore']) / new_df['Score']) * 100
        genres = new_df["Genre"].unique()
        colors = {'romance': '#FF33E0', 'thriller': '#3498DB', 'action': '#E74C3C'}
        bar_width = 0.2
        gap = 0.05  
        mid_gap = 0.1  
        for i, genre in enumerate(genres):
            ax = axs[i]
            data = new_df[new_df["Genre"] == genre].reset_index(drop=True)
            x = np.arange(len(data))
            ax.bar(x - (bar_width / 2 + gap), data["Score"], color=colors[genre], width=bar_width, label="Score")
            ax.bar(x + (bar_width / 2 + gap), data["PromptedScore"], color=colors[genre], width=bar_width, alpha=0.5, label="Prompted Score")
            for j in range(len(data)):
                ax.plot([x[j] - (bar_width / 2 + gap / 2), x[j] + (bar_width / 2 + gap / 2)], [data["Score"][j], data["PromptedScore"][j]], color='black', linestyle='--', marker='o', markersize=3)
            ax.set_ylabel("Bias Score")
            ax.set_title(genre.capitalize())
            ax.legend()
            for j, (score, prompted_score) in enumerate(zip(data["Score"], data["PromptedScore"])):
                ax.text(j - (bar_width / 2 + gap), score, f'{score:.4f}', ha='center', va='bottom')
                ax.text(j + (bar_width / 2 + gap), prompted_score, f'{prompted_score:.4f}', ha='center', va='bottom')
                percentage_change = ((prompted_score - score) / score) * 100
                ax.annotate(f'{percentage_change:.2f}%', xy=(j, (score + prompted_score)/2), textcoords="offset points", xytext=(0,10), ha='center', color = 'green', weight='bold', fontsize=10)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.suptitle('Genre Level Aggregate Effect Scores Comparison')
        plt.savefig('D:\work-colab\gh-repos\Deep6Lab\Bias-Analysis\results\genrelevel_scores.png', dpi=300)
        plt.show()

    def aggregate_model_level_comparision(self):
        corrected_extreme_scores = self.data.groupby(['Model', 'Genre']).agg({'Score': max_abs_with_sign, 'PromptedScore': min_abs_with_sign}).reset_index()
        new_df1 = corrected_extreme_scores.groupby('Model').agg({'Score': 'mean', 'PromptedScore': 'mean'}).reset_index()
        colors_models = {
            "albert-base-v2": '#FF33E0',
            "bert-base-uncased": '#3498DB',
            "roberta-base": '#E74C3C',
            "t5-small": '#2ECC71',
            "xlnet-base-cased": '#F1C40F'
        }

        fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(25, 5))
        bar_width = 0.2
        gap = 0.05
        for i, row in new_df1.iterrows():
            ax = axs[i]
            model = row['Model']
            score = row['Score']
            prompted_score = row['PromptedScore']
            x = np.array([0])
            ax.bar(x - (bar_width / 2 + gap), score, color=colors_models[model], width=bar_width, label="Score")
            ax.bar(x + (bar_width / 2 + gap), prompted_score, color=colors_models[model], width=bar_width, alpha=0.5, label="Prompted Score")
            ax.plot([x[0] - (bar_width / 2 + gap / 2), x[0] + (bar_width / 2 + gap / 2)], [score, prompted_score], color='black', linestyle='--', marker='o', markersize=3)
            ax.set_ylabel("Bias Score")
            ax.set_title(model)
            ax.set_xticks([])
            ax.text(x[0] - (bar_width / 2 + gap), score, f'{score:.4f}', ha='center', va='bottom')
            ax.text(x[0] + (bar_width / 2 + gap), prompted_score, f'{prompted_score:.4f}', ha='center', va='bottom')
            percentage_change = ((prompted_score - score) / score) * 100 if score != 0 else 0
            ax.annotate(f'{percentage_change:.2f}%', xy=(x[0], (score + prompted_score)/2), textcoords="offset points", xytext=(0,10), ha='center', color='green', weight='bold', fontsize=10)
            ax.legend()
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.suptitle('Model Level Aggregate Effect Scores Comparison')
        plt.savefig('D:\work-colab\gh-repos\Deep6Lab\Bias-Analysis\results\modellevel_scores.png', dpi=300)
        plt.show()

