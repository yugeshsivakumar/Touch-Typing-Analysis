import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import os
import zipfile
import datetime
from datetime import timedelta
import matplotlib.pyplot as plt

st.set_page_config(page_title="Touch Typing Analysis", layout="wide")

st.markdown(
    """
    <style>
    .title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        color: #4CAF50;
    }
    .copy-btn {
        background-color: #4CAF50;
        border: none;
        color: white;
        padding: 5px 10px;
        text-align: center;
        font-size: 14px;
        cursor: pointer;
        border-radius: 5px;
    }
    .copy-btn:hover {
        background-color: #45a049;
    }
    .redirect-btn {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 5px 10px;
        text-align: center;
        border-radius: 5px;
        font-size: 14px;
        cursor: pointer;
    }
    .redirect-btn:hover {
        background-color: #45a049;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="title">Touch Typing Analysis</div>', unsafe_allow_html=True)


def generate_dynamic_graph(hours, months, years, predicted_wpm):

    x = np.linspace(0, 24, 100)  
    y = (x * 2) + (months * 1.5) + (years * 3)  
    
    fig, ax = plt.subplots(figsize=(8, 6))  
    fig.patch.set_facecolor('#2C3E50')  
    
    ax.set_facecolor('#34495E')
    
    ax.plot(x, y, label="Predicted WPM", color="cyan", linewidth=2.5, linestyle='-', alpha=0.8)  
    ax.scatter(hours, predicted_wpm, color="lime", label="Predicted WPM (your input)", zorder=5, s=100, edgecolor="black")  
    ax.grid(True, which='both', linestyle='--', linewidth=0.7, color='gray', alpha=0.5)  
    
    ax.set_title("Typing Speed vs Time Spent Typing", fontsize=16, color='white', fontweight='bold')  
    ax.set_xlabel("Cumulative Hours Spent Typing", fontsize=12, color='white')  
    ax.set_ylabel("Predicted WPM (Words per Minute)", fontsize=12, color='white')  
    
    ax.tick_params(axis='x', labelcolor='white', labelsize=10)
    ax.tick_params(axis='y', labelcolor='white', labelsize=10)

    ax.legend(loc="upper left", fontsize=12, facecolor='black', edgecolor='black', framealpha=0.8, labelcolor='white')
    
    for spine in ax.spines.values():
        spine.set_edgecolor('white')
    
    return fig

def calculate_time_saved(standard_speed, user_speed, word_count):
    standard_time = word_count / standard_speed  
    user_time = word_count / user_speed  

    if user_speed < standard_speed:
        time_difference = abs(standard_time - user_time)
        return f"I can save {time_difference:.2f} {'minute' if time_difference == 1 else 'minutes'} than you to complete the paragraph."
    elif user_speed > standard_speed:
        time_difference = abs(user_time - standard_time)
        return f"You can save {time_difference:.2f} {'minute' if time_difference == 1 else 'minutes'} than me to complete the paragraph."
    else:
        return "We both type at the same speed, so no time difference!"


def create_grid_with_code(images_with_data):
    rows = 4
    cols = 4
    idx = 0

    for row in range(rows):
        cols_layout = st.columns(cols)
        for col in range(cols):
            if idx < len(images_with_data):
                img_data = images_with_data[idx]
                with cols_layout[col]:
                    st.image(img_data["image_path"], use_column_width=True, caption=f"Analysis {idx + 1}")
                    
                    with st.expander(f"Code Snippet"):
                        st.code(img_data["code_snippet"], language="python")
                        st.markdown(
                            f'<button class="copy-btn" onclick="navigator.clipboard.writeText({img_data["code_snippet"]})">Copy Code</button>',
                            unsafe_allow_html=True,
                        )
                idx += 1

images_with_data = [
    {"image_path": "Images/1.png", "description": "Description for Image 1", "code_snippet": "plt.figure(figsize=(12, 6))\nsns.lineplot(x=daily_wpm.index, y=daily_wpm.values, marker='o', linewidth=2.5, label='Daily WPM', color='blue')\nplt.xticks(rotation=45, fontsize=10)\nplt.yticks(fontsize=10)\nplt.title('Daily WPM Trend', fontsize=16, fontweight='bold')\nplt.xlabel('Days', fontsize=12)\nplt.ylabel('WPM', fontsize=12)\nplt.legend(fontsize=10)\nplt.tight_layout()\nplt.show()"},
    {"image_path": "Images/2.png", "description": "Description for Image 2", "code_snippet": "plt.figure(figsize=(12, 6))\nsns.lineplot(x=monthly_wpm.index.astype(str), y=monthly_wpm.values, marker='o', linewidth=2.5, label='Monthly WPM', color='green')\nplt.xticks(rotation=45, fontsize=10)\nplt.yticks(fontsize=10)\nplt.title('Monthly WPM Trend', fontsize=16, fontweight='bold')\nplt.xlabel('Month', fontsize=12)\nplt.ylabel('WPM', fontsize=12)\nplt.legend(fontsize=10)\nplt.tight_layout()\nplt.show()"},
    {"image_path": "Images/3.png", "description": "Description for Image 3", "code_snippet": "plt.figure(figsize=(12, 6))\nax = sns.barplot(x=yearly_wpm.index, y=yearly_wpm.values, hue=yearly_wpm.index, dodge=False, palette='coolwarm', edgecolor='black', legend=False)\nfor p in ax.patches:\n    ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', fontsize=20, color='black', xytext=(0, 9), textcoords='offset points')\nplt.xticks(fontsize=10)\nplt.yticks(fontsize=10)\nplt.title('Yearly WPM Trend', fontsize=16, fontweight='bold')\nplt.xlabel('Year', fontsize=12)\nplt.ylabel('WPM', fontsize=12)\nplt.tight_layout()\nplt.show()"},
    
    {"image_path": "Images/4.png", "description": "Description for Image 4", "code_snippet": "plt.figure(figsize=(12, 6))\nsns.boxplot(data=data, x='month', y='wpm', hue='month', dodge=False, palette='Set2', width=0.6, fliersize=5, linewidth=1.2)\nplt.xticks(rotation=45, fontsize=10)\nplt.yticks(fontsize=10)\nplt.title('WPM Distribution by Month', fontsize=16, fontweight='bold')\nplt.xlabel('Month', fontsize=12)\nplt.ylabel('WPM', fontsize=12)\nplt.tight_layout()\nplt.show()"},
    
    {"image_path": "Images/5.png", "description": "Description for Image 5", "code_snippet": "plt.figure(figsize=(15, 5))\nplt.subplot(1, 2, 1)\nsns.lineplot(x=minute_wpm.index, y=minute_wpm.values, marker='o', label='WPM per Minute', color='orange')\nplt.title('WPM Increase per Minute')\nplt.xlabel('Minutes Spend')\nplt.ylabel('Average WPM')\nplt.tight_layout()\nplt.show()"},
    
    {"image_path": "Images/6.png", "description": "Description for Image 6", "code_snippet": "plt.figure(figsize=(15, 5))\nplt.subplot(1, 2, 2)\nsns.lineplot(x=hour_wpm.index, y=hour_wpm.values, marker='o', label='WPM per Hour')\nplt.title('WPM Increase per Hour')\nplt.xlabel('Hours Spend')\nplt.ylabel('Average WPM')\nplt.tight_layout()\nplt.show()"},
    
    {"image_path": "Images/7.png", "description": "Description for Image 7", "code_snippet": "data['acc_group'] = pd.cut(data['acc'], bins=[90, 95, 98, 100], labels=['90-95%', '95-98%', '98-100%'])\nsns.set_theme(style='darkgrid')\nplt.figure(figsize=(10, 6))\nsns.boxplot(data=data, x='acc_group', y='wpm', hue='acc_group', dodge=False, palette='coolwarm', legend=False)\nplt.title('Impact of Accuracy on WPM')\nplt.xlabel('Accuracy')\nplt.ylabel('WPM')\nplt.grid(True)\nplt.show()"},
     {"image_path": "Images/8.png", "description": "Description for Image 8", "code_snippet": "daily_wpm = df.groupby('date')['wpm'].mean().reset_index()\nplt.figure(figsize=(11, 6))\nsns.lineplot(data=daily_wpm, x='date', y='wpm', marker='o', label='Daily Average WPM', color='purple')\nplt.title('Daily WPM Progress')\nplt.xlabel('Date')\nplt.ylabel('Average WPM')\nplt.grid(True)\nplt.xticks(rotation=45)\nplt.legend()\nplt.show()"},
    
    {"image_path": "Images/9.png", "description": "Description for Image 9", "code_snippet": "data['date'] = pd.to_datetime(data['date'])\ndaily_avg_wpm = data.groupby('date')['wpm'].mean().reset_index()\nhighest_wpm_so_far = []\ncurrent_high = -float('inf')\nfor wpm in daily_avg_wpm['wpm']:\n    if wpm > current_high:\n        current_high = wpm\n    highest_wpm_so_far.append(current_high)\ndaily_avg_wpm['highest_wpm_so_far'] = highest_wpm_so_far\nmatching_points = daily_avg_wpm[daily_avg_wpm['wpm'] == daily_avg_wpm['highest_wpm_so_far']]\nmatching_points['days_to_beat'] = matching_points['date'].diff().dt.days\naverage_days_to_beat = matching_points['days_to_beat'].mean()\nplt.figure(figsize=(10, 6))\nsns.lineplot(x=daily_avg_wpm['date'], y=daily_avg_wpm['wpm'], marker='o', color='b', label='Average WPM')\nsns.lineplot(x=daily_avg_wpm['date'], y=daily_avg_wpm['highest_wpm_so_far'], color='r', linestyle='-', label='Highest WPM Progression')\nplt.scatter(matching_points['date'], matching_points['wpm'], color='g', s=50, zorder=5, label='High Score')\nplt.title('WPM vs Date with Highest WPM Progression', fontsize=16)\nplt.xlabel('Date', fontsize=14)\nplt.ylabel('WPM', fontsize=14)\nplt.xticks(rotation=45)\nplt.grid(True)\nplt.text(0.55, 0.9, f'Avg days to beat my own high score: {average_days_to_beat:.2f}', ha='center', va='center', transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=1'))\nplt.legend()\nplt.tight_layout()\nplt.show()"},
    
    {"image_path": "Images/10.png", "description": "Description for Image 10", "code_snippet": "data['acc_group'] = pd.cut(data['acc'], bins=[90, 95, 98, 100], labels=['90-95%', '95-98%', '98-100%'])\nplt.figure(figsize=(11, 6))\nsns.set(style='darkgrid')\nsns.boxplot(data=data, x='acc_group', y='wpm', palette='husl', dodge=False, hue='acc_group', legend=False)\nplt.title('Impact of Accuracy on WPM')\nplt.xlabel('Accuracy')\nplt.ylabel('WPM')\nplt.grid(True)\nplt.show()"},
    
    {"image_path": "Images/11.png", "description": "Description for Image 11", "code_snippet": "consistency_bins = [0, 70, 80, 90, 100]\nconsistency_labels = ['<70', '70-80', '80-90', '90-100']\ndata['consistency_range'] = pd.cut(data['consistency'], bins=consistency_bins, labels=consistency_labels)\nconsistency_wpm = data.groupby('consistency_range', observed=False)['wpm'].mean()\nplt.figure(figsize=(8.3, 5))\nsns.set(style='darkgrid')\nax = sns.barplot(x=consistency_wpm.index, y=consistency_wpm.values, palette='bright', hue=consistency_wpm.index, dodge=False, legend=False)\nfor p in ax.patches:\n    ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', fontsize=12, color='black', xytext=(0, 5), textcoords='offset points')\nplt.title('Average WPM by Consistency Range')\nplt.xlabel('Consistency')\nplt.ylabel('Average WPM')\nplt.tight_layout()\nplt.show()"},
    
    {"image_path": "Images/12.png", "description": "Description for Image 12", "code_snippet": "bins = [4.5, 6, 10, 15, 22]\nlabels = ['4.5-6 min', '6-10 min', '10-15 min', '15-22 min']\ndata['test_duration_range'] = pd.cut(data['testDuration'], bins=bins, labels=labels, include_lowest=True)\nrange_wpm = data.groupby('test_duration_range', observed=False)['wpm'].mean().reset_index()\nplt.figure(figsize=(8.3, 5))\nsns.set(style='darkgrid')\nsns.barplot(data=range_wpm, x='test_duration_range', y='wpm', palette='autumn', hue='test_duration_range', dodge=False, legend=False)\nplt.title('Average WPM by Test Duration')\nplt.xlabel('Test Duration')\nplt.ylabel('Average WPM')\nplt.tight_layout()\nplt.show()"},
    {"image_path": "Images/13.png", "description": "Description for Image 13", "code_snippet": "data_2024 = data[data['year'] == 2024]\nmonthly_avg_wpm = data_2024.groupby('month')['wpm'].mean().reset_index()\nplt.figure(figsize=(8.3, 5))\nsns.barplot(y=monthly_avg_wpm['month'], x=monthly_avg_wpm['wpm'], palette='nipy_spectral', hue=monthly_avg_wpm['month'], dodge=False, legend=False)\nplt.title('Average WPM for Each Month in 2024')\nplt.ylabel('Month')\nplt.xlabel('Average WPM')\nplt.yticks(ticks=range(0, 12), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])\nplt.tight_layout()\nplt.show()\nprint(monthly_avg_wpm.sort_values(by='month', ascending=True))"},
    
    {"image_path": "Images/14.png", "description": "Description for Image 14", "code_snippet": "data['charStats'] = pd.to_numeric(data['charStats'], errors='coerce')\ndata = data.dropna(subset=['charStats'])\ncharstats_bins = [37, 43, 49, 55, 61, 63]\ncharstats_labels = ['37-42', '43-48', '49-54', '55-60', '61-63']\ndata['charstats_range'] = pd.cut(data['charStats'], bins=charstats_bins, labels=charstats_labels, right=True)\nsns.set(style='darkgrid')\npalette = sns.color_palette('Set2', len(charstats_labels))\nplt.figure(figsize=(10, 6))\nsns.boxplot(x='charstats_range', y='wpm', data=data, palette=palette, width=0.6, hue='charstats_range', dodge=False, legend=False)\nplt.xlabel('Character Count', fontsize=14)\nplt.ylabel('WPM', fontsize=14)\nplt.title('WPM Distribution Across Characters count', fontsize=16, fontweight='bold')\nplt.tight_layout()\nplt.show()"},
    
    {"image_path": "Images/15.png", "description": "Description for Image 15", "code_snippet": "sns.set_theme(style='darkgrid')\nlast_10_tests = data.head(10)\ntest_count = range(1, 11)\nmean_wpm = last_10_tests['wpm'].mean()\nlowest_wpm = last_10_tests['wpm'].min()\nhighest_wpm = last_10_tests['wpm'].max()\nplt.figure(figsize=(10, 6))\nsns.lineplot(x=test_count, y=last_10_tests['wpm'], marker='o', color='b', label='WPM')\nplt.axhline(mean_wpm, color='black', linestyle='--', label=f'Mean WPM: {mean_wpm:.2f}')\nplt.axhline(lowest_wpm, color='red', linestyle='--', label=f'Lowest WPM: {lowest_wpm:.2f}')\nplt.axhline(highest_wpm, color='green', linestyle='--', label=f'Highest WPM: {highest_wpm:.2f}')\nplt.title('WPM for the Last 10 Tests', fontsize=16)\nplt.xlabel('Test Number', fontsize=14)\nplt.ylabel('WPM', fontsize=14)\nplt.legend()\nplt.grid(True)\nplt.tight_layout()\nplt.show()"},
    
    {"image_path": "Images/16.png", "description": "Description for Image 16", "code_snippet": "plt.style.use('dark_background')\nplt.figure(figsize=(8.3, 8))\ncolors = plt.cm.Paired(range(len(range_counts)))\nwedges, texts = plt.pie(range_counts, startangle=140, colors=colors, wedgeprops={'edgecolor': 'black', 'linewidth': 0.5}, radius=1.2)\nplt.legend(labels=[f'{label}: {count}' for label, count in zip(range_counts.index, range_counts)], loc='center left', bbox_to_anchor=(1, 0.5), title='WPM Ranges', fontsize=12, facecolor='black', edgecolor='white', title_fontsize=20, labelcolor='white', borderaxespad=3, borderpad=5)\nplt.title('Distribution of WPM', fontsize=20, fontweight='bold', color='white', pad=40)\nplt.subplots_adjust(top=0.85)\nplt.tight_layout()\nplt.show()"}
]
st.markdown("# Pictorial Analysis")
create_grid_with_code(images_with_data)

# Key Metrics Section
st.markdown("---")
df = pd.read_csv('data.csv')
overall_averages = df[['wpm', 'acc', 'rawWpm', 'consistency']].mean()
df['total_time'] = df['testDuration'] + df['incompleteTestSeconds']
Total_sec = df['total_time'].sum()
Total_hrs = str(datetime.timedelta(seconds=int(Total_sec)))


st.markdown("# Key Metrics on Average")
metric1, metric2, metric3, metric4, metric5 = st.columns(5)

with metric1:
    st.metric("WPM", f"{overall_averages['wpm']:.2f}")
with metric2:
    st.metric("Accuracy", f"{overall_averages['acc']:.2f}%")
with metric3:
    st.metric("Consistency", f"{overall_averages['consistency']:.2f}%")
with metric4:
    st.metric("Total hrs spend", f"{Total_hrs} hrs")
with metric5:
    st.metric("Highest WPM", f"{df['wpm'].max()} WPM")

# Streamlit interface - Prediction Section
st.markdown("---")
st.markdown("# My Typing Speed Prediction")

with open('Model/wpm_model_xgb.pkl', 'rb') as f:
    model = pickle.load(f)

df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

# Feature engineering: Convert timestamp to numerical format (e.g., days since the first entry)
df['days_since_start'] = (df['timestamp'] - df['timestamp'].min()).dt.days

# Calculate the current average WPM
current_avg_wpm = df['wpm'].mean()

left_column, right_column = st.columns(2)

with left_column:
    # Sliders for input
    days = st.slider('Days', 0, 365, 0)
    months = st.slider('Months', 0, 12, 0)
    years = st.slider('Years', 0, 10, 0)

    total_days = days + (months * 30) + (years * 365)

    # Prediction logic
    if total_days == 0:
        predicted_wpm = current_avg_wpm
        future_date = pd.to_datetime("today")
    else:
        # Prepare input for prediction
        X_predict = pd.DataFrame({'days_since_start': [total_days]})
        predicted_change = model.predict(X_predict)[0]
        predicted_wpm = max(current_avg_wpm + predicted_change, 0)  # Avoid negative WPM
        future_date = pd.to_datetime("today") + timedelta(days=total_days)

    st.write(f"Predicted WPM for {future_date.strftime('%Y-%m-%d')}: {predicted_wpm:.2f}")

    st.markdown(
    """
    <div style="background-color: #8B0000; padding: 35px; border-radius: 5px; margin-top: 20px;">
        <h4 style="color: white; text-align: center;">Things to Avoid in Touch Typing</h4>
        <ul style="color: white; font-size: 14px; margin-left: 20px;">
            <li>Avoid looking at the keyboard.</li>
            <li>Maintain good posture.</li>
            <li>Use all fingers for typing.</li>
            <li>Ensure proper finger placement.</li>
            <li>Prioritize accuracy over speed.</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True
)
with right_column:
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot historical data
    ax.plot(df['timestamp'], df['wpm'], label='Historical WPM', color='blue', marker='o')

    # Add prediction point
    if total_days > 0:
        ax.scatter(future_date, predicted_wpm, color='red', label=f'Predicted WPM', s=100)

    # Customize the plot
    ax.set_xlabel('Date')
    ax.set_ylabel('WPM')
    ax.set_title('Historical WPM vs Predicted WPM')
    ax.legend()

    plt.xticks(rotation=45)
    plt.tight_layout()

    st.pyplot(fig)

st.markdown("---")

# Header

st.markdown("# Additional Options")

row = st.columns([3, 3, 2]) 

box_content = """
<div style="background-color: #004d00; padding: 46px; border-radius: 5px; margin-top: 20px;">
    <h4 style="color: white; text-align: center;">Tips to Improve Your Typing Speed</h4>
    <ul style="color: white; font-size: 14px; margin: 10px 0 0 20px;">
        <li>Practice regularly with typing tests.</li>
        <li>Focus on accuracy before speed.</li>
        <li>Use all your fingers for typing.</li>
        <li>Maintain good posture while typing.</li>
        <li>Take short breaks to avoid fatigue.</li>
    </ul>
</div>
"""

download_button_shown = False

with row[0]:
    st.title("Your Speed Analysis")
    uploaded_file = st.file_uploader(
        "Upload your CSV file from the Monkeytype website for your analysis.", 
        type=["csv"]
    )

    if uploaded_file and st.button("OK", key="upload_ok"):
        data_ty_path = "data_ty.csv"
        with open(data_ty_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        analysis_file = "Analysis.ipynb"
        requirements_file = "requirements.txt"

        if os.path.exists(analysis_file):
            zip_path = "Typinganalysis.zip"
            with zipfile.ZipFile(zip_path, "w") as zipf:
                zipf.write(data_ty_path)
                zipf.write(analysis_file)
                if os.path.exists(requirements_file):
                    zipf.write(requirements_file)

            box_content = """
            <div style="background-color: #144ede; padding: 46px; border-radius: 5px; margin-top: 20px;">
                <h4 style="color: white; text-align: center;">How to Analyze Your Typing Speed</h4>
                <ul style="color: white; font-size: 14px; margin: 10px 0 0 20px;">
                    <li>Download the ZIP file.</li>
                    <li>Install the required packages locally or upload the ZIP to Google Colab.</li>
                    <li>In the `Analysis.ipynb` file, change the dataset name from `data.csv` to `data_ty.csv`.</li>
                    <li>Run all cells to view your results.</li>
                </ul>
            </div>
            """

            download_button_shown = True

        else:
            st.error("The file 'analysis.ipynb' is missing in the directory.")

    st.markdown(box_content, unsafe_allow_html=True)

    if download_button_shown:
        st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True) 
        with open(zip_path, "rb") as zip_file:
            st.download_button(
                label="Download Typing Analysis",
                data=zip_file,
                file_name="Typinganalysis.zip",
                mime="application/zip"
            )

with row[1]:
    st.title("Time Difference Comparison")
    
    standard_speed = overall_averages['wpm']

    typing_speed = st.text_input("Enter your typing speed (WPM):", "1")
    word_count = st.text_input("Enter the number of words in the paragraph:", "100")

    if st.button("Compare difference on time taken"):
        try:
            typing_speed = float(typing_speed)
            word_count = float(word_count)

            if typing_speed <= 0 or word_count <= 0:
                st.error("Please enter positive numbers for typing speed and word count.")
            else:
                result = calculate_time_saved(standard_speed, typing_speed, word_count)
                st.success(result)
        except ValueError:
            st.error("Please enter valid numeric values for typing speed and word count.")

with row[2]:
    st.title("Medium Post")
    st.image("Images/thumbnail1.jpg", width=450) 
    button_1 = st.button("Medium link", key="redirect_1")
    if button_1:
        st.markdown('<meta http-equiv="refresh" content="0;URL=\'https://medium.com/@Yugesh_S\'">', unsafe_allow_html=True)

# Footer Quote
st.markdown(
    """
    <div style="background: linear-gradient(135deg, #c9f9ff, #93d5fc); padding: 20px; border-radius: 10px; margin-top: 20px; border: 2px solid #5ac8fa; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);">
        <p style="font-size: 15px; color: #00334d; font-weight: bold; text-align: center;">
            <span style="color: red; font-size: 20px;">*</span> 
            My regular work patterns may help me improve my typing speed to some extent. 
            However, your practice routine and individual conditions may vary, potentially making you faster than me in certain situations 
            or requiring fewer hours of practice. Keep in mind that our progress is unique, so it’s not about comparing my speed with yours.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; font-size: 18px; font-style: italic; color: #555;">
        "Touch typing is not just a skill; it's a superpower that transforms thoughts into words at the speed of inspiration."
    </div>
    """,
    unsafe_allow_html=True,
)
