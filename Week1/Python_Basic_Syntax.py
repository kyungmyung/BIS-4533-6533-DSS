# Basic Commands
# 1.Variables
# 1.1. Numerical variables
num = 2
print(num) # Int_Type
num = num * 2.0 # Float_type
print(num)

# 1.2. Strings
string1 = 'Press'
string2 = 'Button'
print(string1)
print(string2)
string3 = string1 +' '+ string2
print(string3)

# 1.3. Tuples
data = ('Student1','2020', "(1990,01,01)")
data[0]
data[1]
data[2]
# Get Error Tuples are immutable
data[0] = "Student2"

# 1.4. Lists
listnum = [1,2,3]
print(listnum)
listnum.append(4)
print(listnum)

# 1.5. Matrix
mat1 = [[1,2,3],
        [4,5,6],
        [7,8,9]]
#print first row
print(mat1[0])
print(mat1[1][2])

# 1.6. Arithmetic Operators
print(3+3)
print(3-3)
print(3*3)
print(3/3)
print(3**3)
print(3%3)

# 1.7. Comparison
value1 = 10
value2 = 20
print(value1<value2)
print(value1<=value2)
print(value1>value2)
print(value1>=value2)
print(value1==value2)
print(value1!=value2)


# get packages
import pandas as pd
# Check current path
os.getcwd()

# 1.1. File_reading and File_writing
data = pd.read_csv('yourpath/filename.csv')
# Check data labels
data.columns
# Check Rows and Columns
data.shape
# Check information of variable types
data.info()
# Check number of observations, mean, std, min, max
data.describe()
des = data.describe()
# saving descriptive statistics
des.to_csv('yourpath/filename.csv')
# Check number of observations, mean, std, min, max of specific variable
data['housing_median_age'].describe()

# 1.2. Data Labeling
long = data['longitude']
lat = data['latitude']
long = long.astype(str)
lat = lat.astype(str)
label = long +' '+ lat
# Covert label as data frame
label = pd.DataFrame(label)
# Change columns name as label
label.columns = ['label']

# 1.3. Data Combining
# combining data
newdata = pd.concat([data, label], axis=1)
# getting unique data label(long, lati)
len(pd.Series.unique(newdata['label']))
len(pd.Series.unique(newdata['longitude']))
len(pd.Series.unique(newdata['latitude']))

# 1.1. if conditions
value = 10
if value>10:
    print("Value > 10 :", value)
if value<=10:
    print("Value <= 10 :", value)
if value>10:
    print("Value < 10: ", value)
elif value<=10:
    print("Value <= 10 :", value)

# 1.2. Loops
max = 5
for i in range(max):
    print(i)
for i in range(max):
    print(i)
    if i == 2:
        break

# 1.3. Functions
def sqaure(value):
    value = value**2
    return value

d1 = sqaure(10)
d2 = sqaure(20)

newvalue = sqaure(value)
print(value)

# 1.4. Error control
try:
    di = 12/0
except:
    print('Division by Zero')

print(di)















