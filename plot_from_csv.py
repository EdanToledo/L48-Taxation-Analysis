import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

all_num_brackets = [3, 7, 50]

coin = True
for metric in ["Coin", "Labour", "Income Equality"]:
    plt.figure(figsize=(40, 6))

    for i, num_brackets in enumerate(all_num_brackets):
        plt.subplot(1, 3, i+1)

        country = "Iceland" if num_brackets == 3 else "US"


        base_path = (
            "/Users/edantoledo/University Work/MLandthePhysicalWorld/FinalProject/other_data_csv"
        )

        free_market_csv_path = f"{base_path}/free_market_{num_brackets}.csv"
        optimised_taxes_csv_path = f"{base_path}/opt_{num_brackets}.csv"
        saez_taxes_csv_path = f"{base_path}/{num_brackets}_saez.csv"

        country_taxes_csv_path = f"{base_path}/{country}.csv"

        free_market_df = pd.read_csv(free_market_csv_path)
        saez_taxes_df = pd.read_csv(saez_taxes_csv_path)
        optimised_taxes_df = pd.read_csv(optimised_taxes_csv_path)
        country_taxes_df = pd.read_csv(country_taxes_csv_path)
        
        free_market_df["Step"] = free_market_df["Step"]*(50//9)
        saez_taxes_df["Step"] = saez_taxes_df["Step"]*(50//9)
        optimised_taxes_df["Step"] = optimised_taxes_df["Step"]*(50//9)
        country_taxes_df["Step"] = country_taxes_df["Step"]*(50//9)

        sns.lineplot(data=free_market_df, x="Step", y=metric, label="Free Market")
        sns.lineplot(
            data=optimised_taxes_df, x="Step", y=metric, label="Optimised Tax Brackets"
        )
        sns.lineplot(data=saez_taxes_df, x="Step", y=metric, label="Saez Taxes")
        if num_brackets == 3 or num_brackets == 7:
            sns.lineplot(data=country_taxes_df, x="Step", y=metric, label=f"{country} Taxes")

        
        plt.ylim(0, 4500 if metric != "Income Equality" else 1.0)
        plt.ylabel("Total Coin" if metric=="Coin" else "Total Labour" if metric=="Labour" else metric)
        plt.xlabel("Training Episodes (1e3)")
        plt.xticks(range(0,50 ,50//9))
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
