# Define a dictionary to keep track of the counts for each combination
x_counts = {}
y_counts = {}
z_counts = {}

# Open the file and read each line
with open('temp.txt', 'r') as file:
    for line in file:
        _, x, y, z = line.strip().split(' ')
        # Increment the count for the X, Y, and Z values in their respective dictionaries
        if x in x_counts:
            x_counts[x] += 1
        else:
            x_counts[x] = 1

        if y in y_counts:
            y_counts[y] += 1
        else:
            y_counts[y] = 1

        if z in z_counts:
            z_counts[z] += 1
        else:
            z_counts[z] = 1

# Print the counts for each value of X, Y, and Z
print("Counts for X:")
for x, count in x_counts.items():
    print(f"{x}: {count}")

print("Counts for Y:")
for y, count in y_counts.items():
    print(f"{y}: {count}")

print("Counts for Z:")
for z, count in z_counts.items():
    print(f"{z}: {count}")