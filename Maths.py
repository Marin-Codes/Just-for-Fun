import matplotlib.pyplot as plt

# Fields
fields = [
    "Engineering",
    "Natural Sciences",
    "Technology & CS",
    "Finance & Business",
    "Medicine & Healthcare",
    "Everyday Life",
    "Fashion & Design",
    "Art & Music"
]

importance = [95, 90, 85, 80, 65, 60, 50, 40]

colors = []
for value in importance:
    if value >= 90:
        colors.append("red")        
    elif value >= 70:
        colors.append("orange")    
    elif value >= 50:
        colors.append("gold")     
    else:
        colors.append("green")      

# Create bar chart
plt.figure(figsize=(10,6))
bars = plt.bar(fields, importance, color=colors)

# Add percentage labels on top of bars
for i in range(len(bars)):
    plt.text(i, importance[i] + 2, str(importance[i]) + "%", ha='center')

plt.xlabel("Different Fields")
plt.ylabel("Importance of Mathematics (%)")
plt.title("Importance of Mathematics Across Different Fields")

plt.xticks(rotation=45)
plt.ylim(0, 100)

plt.tight_layout()
plt.show()
