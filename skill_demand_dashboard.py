import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import censusdata
import pydeck as pdk

sns.set_theme(style="whitegrid")
st.set_page_config(layout="wide")


def load_data():
    """
    Load datasets related to employment, wages, CTE programs, outcomes, and skill needs.
    Returns seven pandas DataFrames.
    """
    employment_projections = pd.read_csv("data/Employment_Projections_by_Industry.csv", header=1)
    nc_occupational_employment_wages = pd.read_csv("data/NC_Occupational_Employment_and_Wages.csv")
    clean_cte = pd.read_csv("data/3_4_cleanCTE2.csv")
    fifth_cleaned = pd.read_csv("data/5th_cleaned.csv")
    nc_occupational_outcomes = pd.read_csv("data/NC_Occupational_Outcomes_1_Year_After_Grad.csv", header=1)
    nc_post_grad_enrollment = pd.read_csv("data/NC_Post_Grad_Enrollment_After_1_Year.csv", header=1)
    industry_skills_needs = pd.read_csv("data/Industry_Skills_Needs.csv")
    return employment_projections, nc_occupational_employment_wages, clean_cte, fifth_cleaned, nc_occupational_outcomes, nc_post_grad_enrollment, industry_skills_needs

def preprocess_df1(employment_projections):
    """
    Preprocess the employment projections dataset: clean symbols, convert to numeric, handle percentages.
    """
    employment_projections.replace('*', np.nan, inplace=True)
    columns_to_convert = ['2021', '2030', 'Net Growth', 'Average Weekly', 'Annualized']
    for column in columns_to_convert:
        if column in ['Average Weekly', 'Annualized']:
            employment_projections[column] = pd.to_numeric(
                employment_projections[column].astype(str).str.replace(',', '').str.replace('$', '').str.replace('%', ''),
                errors='coerce'
            ) / (100 if column == 'Annualized' else 1)
        else:
            employment_projections[column] = pd.to_numeric(
                employment_projections[column].astype(str).str.replace(',', ''),
                errors='coerce'
            )
    return employment_projections



def plot_average_weekly_wages(df):
    """
    Plots a histogram of average weekly wages to show wage distribution across industries.
    """
    fig, ax = plt.subplots(figsize=(12,6))
    sns.histplot(df['Average Weekly'].dropna(), kde=True, color="cornflowerblue", ax=ax)
    ax.set_title('Distribution of Average Weekly Wages')
    ax.set_xlabel('Average Weekly Wage')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)
    
def plot_top_20_net_growth(df):
    """
    Plots the top 20 industries by projected net job growth.
    """
    top_20_net_growth = df.sort_values(by='Net Growth', ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(12,6))
    sns.barplot(x='Net Growth', y='Industry Title', data=top_20_net_growth, color="cornflowerblue", ax=ax)
    ax.set_title('Top 20 Industries by Net Growth')
    ax.set_xlabel('Net Growth')
    ax.set_ylabel('Industry Title')
    fig.tight_layout()
    st.pyplot(fig)

def plot_top_n_net_growth_in(df):
    """
    Interactive: Plots the top N industries by net growth based on user-defined selection (5–30).
    """
    n = st.sidebar.slider('Select the number of top industries', min_value=5, max_value=30, value=20)
    top_n_net_growth = df.sort_values(by='Net Growth', ascending=False).head(n)
    fig, ax = plt.subplots(figsize=(12,6))
    sns.barplot(x='Net Growth', y='Industry Title', data=top_n_net_growth, color="cornflowerblue", ax=ax)
    ax.set_title(f'Top {n} Industries by Net Growth')
    ax.set_xlabel('Net Growth')
    ax.set_ylabel('Industry Title')
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    st.pyplot(fig)

def interactive_avg_weekly_wages(df):
    """
    Interactive: Plots a histogram of average weekly wages filtered by a user-selected wage range.
    """
    min_wage, max_wage = df['Average Weekly'].min(), df['Average Weekly'].max()
    selected_range = st.sidebar.slider(
        'Select Average Weekly Wage Range',
        min_value=int(min_wage),
        max_value=int(max_wage),
        value=(int(min_wage), int(max_wage))
    )
    filtered_df = df[(df['Average Weekly'] >= selected_range[0]) & (df['Average Weekly'] <= selected_range[1])]
    fig, ax = plt.subplots(figsize=(12,6))
    sns.histplot(filtered_df['Average Weekly'], kde=True, ax=ax)
    ax.set_title('Distribution of Average Weekly Wages Within Selected Range')
    ax.set_xlabel('Average Weekly Wage')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)


def show_bubble_chart(df):
    """
    Interactive: Displays a bubble chart showing occupations sized by mean annual wage and positioned by employment.
    """
    df_filtered = df[df['Occupation'] != 'Total, All Occupations'].copy()
    df_filtered['Employment'] = pd.to_numeric(df_filtered['Employment'].str.replace(',', ''), errors='coerce')
    df_filtered['Annual wage; mean'] = pd.to_numeric(df_filtered['Annual wage; mean'].str.replace('[\$,]', '', regex=True), errors='coerce')

    sort_criteria = st.sidebar.selectbox('Sort by:', ['Employment', 'Annual wage; mean'])
    num_occupations = st.sidebar.slider('Number of Occupations to Display', min_value=5, max_value=50, value=25)
    top_occupations = df_filtered.nlargest(num_occupations, sort_criteria)

    fig, ax = plt.subplots(figsize=(12,6))
    bubble_chart = sns.scatterplot(
        data=top_occupations,
        y='Occupation',
        x='Employment',
        size='Annual wage; mean',
        sizes=(100, 2000),
        alpha=0.5,
        ax=ax,
        legend='brief'
    )
    ax.set_title(f'Top {num_occupations} Occupations by {sort_criteria} (Bubble Chart)')
    ax.set_xlabel('Employment')
    ax.set_ylabel('Occupation')
    plt.setp(ax.get_xticklabels(), rotation=90)
    ax.legend(
        title='Annual Wage Mean',
        loc='center left', bbox_to_anchor=(1, 0.8),
        fontsize='large', title_fontsize='x-large'
    )
    ax.grid(True)
    fig.tight_layout()
    st.pyplot(fig)


def interactive_employment_outcomes_by_industry(df):
    """
    Interactive: Line chart showing employment outcomes over time for selected industries.
    """
    df[df.columns[4:]] = df[df.columns[4:]].replace(',', '', regex=True).astype(float)
    industries = st.sidebar.multiselect('Select Industries', options=df.columns[4:], default=[df.columns[4]])
    fig, ax = plt.subplots(figsize=(12,6))
    for column in industries:
        ax.plot(df['Year'], df[column], marker='o', label=column)
        ax.text(df['Year'].iloc[-1], df[column].iloc[-1], column, ha='left', va='center')
    ax.set_title('Employment Outcomes by Industry for Recent Graduates')
    ax.set_xlabel('Year')
    ax.set_ylabel('Employment Count')
    ax.set_xticks(df['Year'])
    ax.set_xticklabels(df['Year'], rotation=45)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    fig.tight_layout()
    st.pyplot(fig)

def interactive_priority_ranking_heatmap(df):
    """
    Interactive: Heatmap showing priority ranking of skill group categories by industry and year.
    """
    skill_categories = ["Specialized Industry Skills", "Business Skills", "Tech Skills", "Soft Skills", "Disruptive Tech Skills"]
    filtered_df = df[df['skill_group_category'].isin(skill_categories)]

    grouped_sums = filtered_df.groupby(
        ['year', 'isic_section_name', 'industry_name', 'skill_group_category']
    )['skill_group_rank'].sum().reset_index(name='sum_skill_group_rank')

    industry_names = st.sidebar.multiselect('Select Industry', options=grouped_sums['industry_name'].unique(), default=grouped_sums['industry_name'].unique()[0])
    selected_year = st.sidebar.slider('Select Year', int(grouped_sums['year'].min()), int(grouped_sums['year'].max()), int(grouped_sums['year'].min()))

    specific_group = grouped_sums[
        (grouped_sums['industry_name'].isin(industry_names)) & (grouped_sums['year'] == selected_year)
    ]

    specific_group['quantile_rank'] = pd.qcut(specific_group['sum_skill_group_rank'], 5, labels=False) + 1
    specific_group['priority_rank'] = specific_group['quantile_rank'].apply(lambda x: 6 - x)

    heatmap_data = specific_group.pivot(index='skill_group_category', columns='year', values='priority_rank')

    fig, ax = plt.subplots(figsize=(12,6))
    sns.heatmap(
        heatmap_data,
        annot=True,
        cmap='inferno',
        fmt=".0f",
        linewidths=.5,
        cbar_kws={'label': 'Priority Rank (1=Highest, 5=Lowest)'},
        ax=ax
    )
    ax.set_title('Priority Ranking Heatmap of Skill Group Categories by Year')
    ax.set_ylabel('Skill Group Category')
    ax.set_xlabel('Year')
    plt.setp(ax.get_xticklabels(), rotation=45)
    fig.tight_layout()
    st.pyplot(fig)

def interactive_priority_ranking_group(df):
    """
    Interactive: Bar chart showing the count of different skill group categories within a selected industry.
    """
    industry_name = st.sidebar.selectbox('Select Industry', options=df['industry_name'].unique())
    filtered_df = df[df['industry_name'] == industry_name]
    skill_group_distribution = filtered_df.groupby('skill_group_category').size().reset_index(name='Count')
    fig, ax = plt.subplots(figsize=(12,6))
    sns.barplot(x='Count', y='skill_group_category', data=skill_group_distribution, palette='coolwarm', ax=ax)
    ax.set_title(f'Skill Group Distribution in {industry_name}')
    ax.set_xlabel('Count')
    ax.set_ylabel('Skill Group Category')
    fig.tight_layout()
    st.pyplot(fig)

def interactive_top_skill_groups(df):
    """
    Interactive: Displays the most common skill groups across all industries by count.
    """
    skill_group_counts = df['skill_group_name'].value_counts().reset_index(name='Count')
    skill_group_counts.columns = ['Skill Group', 'Count']
    num_skill_groups = st.sidebar.slider('Number of Top Skill Groups to Display', 1, len(skill_group_counts), 10)
    fig, ax = plt.subplots(figsize=(12,6))
    sns.barplot(x='Count', y='Skill Group', data=skill_group_counts.head(num_skill_groups), palette='viridis', ax=ax)
    ax.set_title(f'Top {num_skill_groups} Skill Groups Across Industries')
    ax.set_xlabel('Count')
    ax.set_ylabel('Skill Group')
    fig.tight_layout()
    st.pyplot(fig)


def main():
    st.title("Skill Demand Insights")

    df1, df2, df34, df5, df6, df7, df8 = load_data()

    dataset_name = st.sidebar.selectbox("Select Dataset", (
        "Employment Projections",
        "Occupational Employment and Wages",
        "Employment Outcomes",
        "Top Skill Groups",
        "Skill Group Heatmap",
        "Skill Group Distribution",
    ))

    if dataset_name == "Employment Projections":
        df1 = preprocess_df1(df1)
        visualization = st.sidebar.selectbox(
            "Select a Visualization",
            [
                "Average Weekly Wages",
                "Top 20 Industries by Net Growth",
                "Top N Industries by Net Growth",
                "Average weekly N wage"
            ]
        )
        if visualization == "Average Weekly Wages":
            plot_average_weekly_wages(df1)
        elif visualization == "Top 20 Industries by Net Growth":
            plot_top_20_net_growth(df1)
        elif visualization == "Top N Industries by Net Growth":
            plot_top_n_net_growth_in(df1)
        elif visualization == "Average weekly N wage":
            interactive_avg_weekly_wages(df1)

    elif dataset_name == "Occupational Employment and Wages":
        show_bubble_chart(df2)

    elif dataset_name == "Employment Outcomes":
        interactive_employment_outcomes_by_industry(df6)

    elif dataset_name == "Skill Group Heatmap":
        interactive_priority_ranking_heatmap(df8)

    elif dataset_name == "Skill Group Distribution":
        interactive_priority_ranking_group(df8)

    elif dataset_name == "Top Skill Groups":
        interactive_top_skill_groups(df8)

if __name__ == "__main__":
    main()
