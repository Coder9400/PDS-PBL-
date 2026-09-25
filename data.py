# ==========================================================  
# Employee Salary Analysis - Micro Project  
# Libraries: Python, Pandas, NumPy, Matplotlib  
# ========================================================== 
import numpy as np 
import pandas as pd 
import matplotlib 
matplotlib.use("Agg")  
import matplotlib.pyplot as plt  
   
# ---------- 1. Create the dataset ---------- np.random.seed(42) 
n = 60  
departments = ["HR", "IT", "Finance", "Marketing", "Sales"] 
genders = ["Male", "Female"]  
   
df = pd.DataFrame({  
    "EmpID": range(101, 101 + n),  
    "Name": [f"Employee_{i}" for i in range(1, n + 1)],  
    "Department": np.random.choice(departments, n),  
    "Gender": np.random.choice(genders, n),  
    "Experience": np.random.randint(1, 21, n),       
    # years 
})  
   
base = {"HR": 25000, "IT": 40000, "Finance": 35000,  
        "Marketing": 30000, "Sales": 28000} 
df["Salary"] = (df["Department"].map(base)  
                + df["Experience"] * 1800  
                + np.random.randint(-4000, 4000, n))  
   
df.to_csv("employee_data.csv", index=False)  
   
# ---------- 2. Explore the data ---------- 
print("=== First 5 Records ===") 
print(df.head())  
print("\n=== Dataset Info ===") 
print("Rows, Columns:", df.shape)  
print("Missing values:\n", df.isnull().sum()) 
print("\n=== Statistical Summary ===") 
print(df[["Experience", "Salary"]].describe().round(2))  
   
# ---------- 3. NumPy statistics ---------- 
sal = df["Salary"].to_numpy()  
print("\n=== NumPy Statistics on Salary ===") 
print("Mean     :", round(np.mean(sal), 2)) 
print("Median   :", np.median(sal)) 
print("Std Dev  :", round(np.std(sal), 2)) 
print("Max      :", np.max(sal)) 
print("Min:", np.min(sal))  
   
# ---------- 4. Pandas analysis ---------- 
print("\n=== Average Salary by Department ===") 
dept_avg = df.groupby("Department")["Salary"].mean().round(2).sort_values(ascending=False) 
print(dept_avg)  
   
print("\n=== Average Salary by Gender ===")  
gender_avg = df.groupby("Gender")["Salary"].mean().round(2) 
print(gender_avg)  
   
print("\n=== Employee Count by Department ===") 
dept_count = df["Department"].value_counts() 
print(dept_count)  
   
print("\n=== Top 5 Highest Paid Employees ===")  
print(df.nlargest(5, "Salary")[["Name", "Department", "Experience", "Salary"]])  
   
# Salary category using NumPy  
df["Category"] = np.where(df["Salary"] >= 60000, "High",  
                  np.where(df["Salary"] >= 45000, "Medium", "Low")) 
print("\n=== Salary Category Count ===") 
print(df["Category"].value_counts())  
   
# Correlation  
corr = df["Experience"].corr(df["Salary"]) 
print("\nCorrelation (Experience vs Salary):", round(corr, 3))  
   
# ---------- 5. Visualization ---------- 
plt.figure(figsize=(7, 4))  
plt.bar(dept_avg.index, dept_avg.values, color="steelblue") 
plt.title("Average Salary by Department") 
plt.xlabel("Department"); 
plt.ylabel("Average Salary (Rs.)") 
plt.tight_layout(); 
plt.savefig("chart1_bar.png", dpi=150); plt.close()  
   
plt.figure(figsize=(7, 4))  
plt.hist(df["Salary"], bins=10, color="seagreen", edgecolor="black") 
plt.title("Salary Distribution")  
plt.xlabel("Salary (Rs.)"); plt.ylabel("Number of Employees")  
plt.tight_layout(); plt.savefig("chart2_hist.png", dpi=150); 
plt.close()  
   
plt.figure(figsize=(7, 4))  
plt.scatter(df["Experience"], df["Salary"], color="darkorange") 
m, c = np.polyfit(df["Experience"], df["Salary"], 1) 
plt.plot(df["Experience"].sort_values(), m * 
df["Experience"].sort_values() + c, "r--") 
plt.title("Experience vs Salary")  
plt.xlabel("Experience (years)"); plt.ylabel("Salary (Rs.)") 
plt.tight_layout(); plt.savefig("chart3_scatter.png", dpi=150); 
plt.close()  
   
plt.figure(figsize=(5, 5))  
plt.pie(dept_count.values, labels=dept_count.index, autopct="%1.1f%%", 
startangle=90)  
plt.title("Employees by Department")  
plt.tight_layout(); plt.savefig("chart4_pie.png", dpi=150); 
plt.close()  
   
print("\nCharts saved successfully.")