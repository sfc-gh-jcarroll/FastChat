import ast
import glob
import pickle

import numpy as np
import pandas as pd
import streamlit as st

from common import page_setup

is_mirror = False # TODO: Read from CLI here.

page_setup(
    title="Leaderboard",
    icon="🏆",
    wide_mode=True,
)

###############################################################################
# Useful functions

@st.cache_resource(ttl="1d")
def read_leaderboard_file():
    leaderboard_table_files = glob.glob("data/leaderboard_table_*.csv")
    leaderboard_table_files.sort(key=lambda x: int(x[-12:-4]))
    leaderboard_table_file = leaderboard_table_files[-1]

    leaderboard_dict = load_leaderboard_table_csv(leaderboard_table_file)
    leaderboard_df = pd.DataFrame.from_dict(leaderboard_dict)

    return leaderboard_df


@st.cache_resource(ttl="1d")
def read_elo_results_file():
    elo_results_files = glob.glob("data/elo_results_*.pkl")
    elo_results_files.sort(key=lambda x: int(x[-12:-4]))
    elo_results_file = elo_results_files[-1]

    arena_dfs = {}
    category_elo_results = {}

    with open(elo_results_file, "rb") as f:
        elo_results = pickle.load(f)

    last_updated_time = None

    if "full" in elo_results:
        last_updated_time = (
            elo_results["full"]["last_updated_datetime"].split(" ")[0])

        for k, cat_name in key_to_category_name.items():
            if k not in elo_results:
                continue
            arena_dfs[cat_name] = elo_results[k]["leaderboard_table_df"]
            category_elo_results[cat_name] = elo_results[k]

    return arena_dfs, category_elo_results


@st.cache_resource(ttl="1d")
def load_leaderboard_table_csv(filename, add_hyperlink=True):
    with open(filename) as f:
        lines = f.readlines()

    heads = [v.strip() for v in lines[0].split(",")]
    rows = []

    for i, line in enumerate(lines[1:]):
        row = [v.strip() for v in line.split(",")]

        for j in range(len(heads)):
            item = {}

            for h, v in zip(heads, row):
                if h == "Arena Elo rating":
                    if v != "-":
                        v = int(ast.literal_eval(v))
                    else:
                        v = np.nan
                elif h == "MMLU":
                    if v != "-":
                        v = round(ast.literal_eval(v) * 100, 1)
                    else:
                        v = np.nan
                elif h == "MT-bench (win rate %)":
                    if v != "-":
                        v = round(ast.literal_eval(v[:-1]), 1)
                    else:
                        v = np.nan
                elif h == "MT-bench (score)":
                    if v != "-":
                        v = round(ast.literal_eval(v), 2)
                    else:
                        v = np.nan
                item[h] = v

        rows.append(item)

    return rows


@st.cache_resource(ttl="1d")
def get_arena_table(arena_df, model_table_df, arena_subset_df=None):
    arena_df = arena_df.sort_values(
        by=["final_ranking", "rating"], ascending=[True, False]
    )
    arena_df["final_ranking"] = recompute_final_ranking(arena_df)
    arena_df = arena_df.sort_values(by=["final_ranking"], ascending=True)

    # sort by rating
    if arena_subset_df is not None:
        # filter out models not in the arena_df
        arena_subset_df = arena_subset_df[arena_subset_df.index.isin(arena_df.index)]
        arena_subset_df = arena_subset_df.sort_values(by=["rating"], ascending=False)
        arena_subset_df["final_ranking"] = recompute_final_ranking(arena_subset_df)
        # keep only the models in the subset in arena_df and recompute final_ranking
        arena_df = arena_df[arena_df.index.isin(arena_subset_df.index)]
        # recompute final ranking
        arena_df["final_ranking"] = recompute_final_ranking(arena_df)

        # assign ranking by the order
        arena_subset_df["final_ranking_no_tie"] = range(1, len(arena_subset_df) + 1)
        arena_df["final_ranking_no_tie"] = range(1, len(arena_df) + 1)
        # join arena_df and arena_subset_df on index
        arena_df = arena_subset_df.join(
            arena_df["final_ranking"], rsuffix="_global", how="inner"
        )
        arena_df["ranking_difference"] = (
            arena_df["final_ranking_global"] - arena_df["final_ranking"]
        )

        arena_df = arena_df.sort_values(
            by=["final_ranking", "rating"], ascending=[True, False]
        )
        arena_df["final_ranking"] = arena_df.apply(
            lambda x: create_ranking_str(x["final_ranking"], x["ranking_difference"]),
            axis=1,
        )

    arena_df["final_ranking"] = arena_df["final_ranking"].astype(str)

    values = []

    for i in range(len(arena_df)):
        row = []
        model_key = arena_df.index[i]
        try:  # this is a janky fix for where the model key is not in the model table (model table and arena table dont contain all the same models)
            model_name = model_table_df[model_table_df["key"] == model_key][
                "Model"
            ].values[0]
            # rank
            ranking = arena_df.iloc[i].get("final_ranking") or i + 1
            row.append(ranking)
            if arena_subset_df is not None:
                row.append(arena_df.iloc[i].get("ranking_difference") or 0)
            # model display name
            row.append(model_name)
            # elo rating
            row.append(round(arena_df.iloc[i]["rating"]))
            upper_diff = round(
                arena_df.iloc[i]["rating_q975"] - arena_df.iloc[i]["rating"]
            )
            lower_diff = round(
                arena_df.iloc[i]["rating"] - arena_df.iloc[i]["rating_q025"]
            )
            row.append(f"+{upper_diff}/-{lower_diff}")
            # num battles
            row.append(round(arena_df.iloc[i]["num_battles"]))
            # Organization
            row.append(
                model_table_df[model_table_df["key"] == model_key][
                    "Organization"
                ].values[0]
            )
            # license
            row.append(
                model_table_df[model_table_df["key"] == model_key]["License"].values[0]
            )
            cutoff_date = model_table_df[model_table_df["key"] == model_key][
                "Knowledge cutoff date"
            ].values[0]
            if cutoff_date == "-":
                row.append("Unknown")
            else:
                row.append(cutoff_date)
            values.append(row)

        except Exception as e:
            print(f"{model_key} - {e}")

    return values


@st.cache_resource(ttl="1d")
def recompute_final_ranking(arena_df):
    # compute ranking based on CI
    ranking = {}
    for i, model_a in enumerate(arena_df.index):
        ranking[model_a] = 1
        for j, model_b in enumerate(arena_df.index):
            if i == j:
                continue
            if (
                arena_df.loc[model_b]["rating_q025"]
                > arena_df.loc[model_a]["rating_q975"]
            ):
                ranking[model_a] += 1
    return list(ranking.values())


###############################################################################
# Prepare data

key_to_category_name = {
    "full": "Overall",
    "coding": "Coding",
    "long_user": "Longer Query",
    "english": "English",
    "chinese": "Chinese",
    "french": "French",
    "no_tie": "Exclude ties",
    "no_short": "Exclude Short Query (< 5 tokens)",
    "no_refusal": "Exclude Refusal",
}

cat_name_to_explanation = {
    "Overall": "Overall questions",
    "Coding": "Coding: whether conversation contains code snippets",
    "Longer Query": "Longer query (>= 500 tokens)",
    "English": "English prompts",
    "Chinese": "Chinese prompts",
    "French": "French prompts",
    "Exclude Ties": "Exclude ties and bothbad",
    "Exclude Short Query (< 5 tokens)": "Exclude short user query (< 5 tokens)",
    "Exclude Refusal": 'Exclude model responses with refusal (e.g., "I cannot answer")',
}

try:
    arena_dfs, category_elo_results = read_elo_results_file()
    leaderboard_df = read_leaderboard_file()
except:
    st.error("Error loading leaderboard data.", icon=":material/chat_error:")


###############################################################################
# Draw app

c, _ = st.columns([4, 2])

with c:
    if is_mirror:
        st.error("""
            This is a mirror of the live leaderboard created and maintained by the [LMSYS
            Organization](https://lmsys.org). Please link to https://leaderboard.lmsys.org for
            citation purposes.
        """, icon=":material/warning:")

    f"""
    **LMSYS [Chatbot Arena](https://lmsys.org/blog/2023-05-03-arena/) is a crowdsourced open platform for
    LLM evals.** We've collected over 800,000 human pairwise comparisons to rank LLMs with the [Bradley-Terry
    model](https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model) and display the model ratings in
    Elo-scale. You can find more details in our [paper](https://arxiv.org/abs/2403.04132).

    Code to recreate leaderboard tables and plots in this
    [notebook](https://colab.research.google.com/drive/1KdwokPjirkTmpO_P1WByFNFiqxWQquwH). You can
    contribute your vote at [chat.lmsys.org](https://chat.lmsys.org).

    **Rank (UB)**: model's ranking (upper-bound), defined by one + the number of models that are
    statistically better than the target model. Model A is statistically better than model B when
    A's lower-bound score is greater than B's upper-bound score (in 95% confidence interval). See
    Figure 1 below for visualization of the confidence intervals of model scores.
    """

    st.success("""
        Chatbot arena is dependent on community participation, **please contribute by casting your vote!**
    """, icon=":material/person_raised_hand:")

st.write("") # Vertical spacing
st.write("") # Vertical spacing

arena_results, full_leaderboard = st.tabs(["Arena results", "Full leaderboard"])

with arena_results:
    category_name = st.selectbox(
        "Category",
        arena_dfs.keys(),
        help="\n".join(
            f"* **{name}:** {exp}" for name, exp in cat_name_to_explanation.items()
        ))

    arena_df = arena_dfs["Overall"]
    arena_subset_df = arena_dfs[category_name]

    total_votes = sum(arena_df["num_battles"]) // 2
    total_models = len(arena_df)
    total_subset_votes = sum(arena_subset_df["num_battles"]) // 2
    total_subset_models = len(arena_subset_df)

    cols = st.columns(2)

    with cols[0]:
        with st.container(border=True):
            st.metric(
                f'Number of models in "{category_name}"',
                f"{total_subset_models:,} ({round(total_subset_models/total_models *100)}%)",
            )

    with cols[1]:
        with st.container(border=True):
            st.metric(
                f'Number of votes in "{category_name}"',
                f"{total_subset_votes:,} ({round(total_subset_votes/total_votes * 100)}%)",
            )

    arena_table_vals = get_arena_table(arena_subset_df, leaderboard_df)

    arena_vals = pd.DataFrame(
        arena_table_vals,
        columns=[
            "Rank* (UB)",
            "Model",
            "Arena Elo",
            "95% CI",
            "Votes",
            "Organization",
            "License",
            "Knowledge Cutoff",
        ],
    )

    def progress_column(df, col):
        min_value = int(df[col].min())
        max_value = int(df[col].max())
        return st.column_config.ProgressColumn(format="%f", min_value=min_value, max_value=max_value)

    st.dataframe(
        arena_vals,
        height=800,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Arena Elo": progress_column(arena_vals, "Arena Elo"),
            "Votes": progress_column(arena_vals, "Votes"),
        },
    )


st.write("") # Vertical spacing
st.write("") # Vertical spacing


# Plotly Figures.

F"""
## More Statistics for Chatbot Arena - {category_name}
"""

st.write("") # Vertical spacing

cols = st.columns(2)

with cols[0]:
    """
    ##### Figure 1: Fraction of Model A Wins

    :gray[Does not include ties.]
    """
    st.plotly_chart(
        category_elo_results[category_name]["win_fraction_heatmap"],
        use_container_width=True)

with cols[1]:
    """
    ##### Figure 2: Battle Count for Each Combination of Models

    :gray[Does not include ties.]
    """
    st.plotly_chart(
        category_elo_results[category_name]["battle_count_heatmap"],
        use_container_width=True)

st.write("") # Vertical spacing

cols = st.columns(2)

with cols[0]:
    """
    ##### Figure 3: Confidence Intervals on Model Strength

    :gray[Via bootstrapping.]
    """
    st.plotly_chart(
        category_elo_results[category_name]["bootstrap_elo_rating"],
        use_container_width=True)

with cols[1]:
    """
    ##### Figure 4: Average Win Rate Against All Other Models

    :gray[Assumes uniform sampling and no ties]
    """
    st.plotly_chart(
        category_elo_results[category_name]["average_win_rate_bar"],
        use_container_width=True)


c, _ = st.columns([4, 2])

with c:
    st.warning("""
        **Note:** In each category, we exclude models with fewer than 500 votes as their confidence
        intervals can be large.
        """, icon=":material/sticky_note_2:")

    st.write("") # Vertical spacing
    st.write("") # Vertical spacing

    """
    ## How to cite these results

    Please cite the following paper if you find our leaderboard or dataset helpful.

    ```
    @misc{chiang2024chatbot,
        title={Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference},
        author={Wei-Lin Chiang and Lianmin Zheng and Ying Sheng and Anastasios Nikolas Angelopoulos and Tianle Li and Dacheng Li and Hao Zhang and Banghua Zhu and Michael Jordan and Joseph E. Gonzalez and Ion Stoica},
        year={2024},
        eprint={2403.04132},
        archivePrefix={arXiv},
        primaryClass={cs.AI}
    }
    ```
    """
