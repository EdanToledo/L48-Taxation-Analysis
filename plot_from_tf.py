import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

all_num_brackets = [3, 7, 50]


plt.figure(figsize=(40, 6))

for i, num_brackets in enumerate(all_num_brackets):
    plt.subplot(1, 3, i+1)

    country = "Iceland" if num_brackets == 3 else "US"


    base_path = (
        "/Users/edantoledo/University Work/MLandthePhysicalWorld/FinalProject/data_csvs"
    )

    free_market_csv_path = f"{base_path}/FreeMarket{num_brackets}Brackets.csv"
    optimised_taxes_csv_path = f"{base_path}/Optimised{num_brackets}Brackets.csv"
    saez_taxes_csv_path = f"{base_path}/Saez{num_brackets}Brackets.csv"

    country_taxes_csv_path = f"{base_path}/{country}Taxes.csv"

    free_market_df = pd.read_csv(free_market_csv_path)
    saez_taxes_df = pd.read_csv(saez_taxes_csv_path)
    optimised_taxes_df = pd.read_csv(optimised_taxes_csv_path)
    country_taxes_df = pd.read_csv(country_taxes_csv_path)


    

    sns.lineplot(data=free_market_df, x="Step", y="Value", label="Free Market")
    sns.lineplot(
        data=optimised_taxes_df, x="Step", y="Value", label="Optimised Tax Brackets"
    )
    sns.lineplot(data=saez_taxes_df, x="Step", y="Value", label="Saez Taxes")
    if num_brackets == 3 or num_brackets == 7:
        sns.lineplot(data=country_taxes_df, x="Step", y="Value", label=f"{country} Taxes")

    plt.yticks(range(0,700,50))
    plt.ylim(0, 650)
    plt.ylabel("Mean Social Welfare")
    plt.title(f"{num_brackets} Brackets")

    plt.legend()


plt.show()

# free_market_csv_path = '../MLandthePhysicalWorld/FinalProject/free-market-agents.csv'
# us_taxes_csv_path = '../MLandthePhysicalWorld/FinalProject/us-taxes-agents.csv'
# saez_taxes_csv_path = '../MLandthePhysicalWorld/FinalProject/saez-taxes-agents.csv'
# free_market_df = pd.read_csv(free_market_csv_path)
# us_taxes_df = pd.read_csv(us_taxes_csv_path)
# saez_taxes_df = pd.read_csv(saez_taxes_csv_path)


# plt.figure(figsize=(16, 6))

# sns.lineplot(data=free_market_df, x="Step", y="Value", label = "Free Market Agents")
# sns.lineplot(data=us_taxes_df, x="Step", y="Value" , label = "US Taxes Agents")
# sns.lineplot(data=saez_taxes_df, x="Step", y="Value", label = "Saez Taxes Agents")
# plt.title("Mean Agent Reward (Marginal Utility)")
# plt.legend()
# plt.show()
