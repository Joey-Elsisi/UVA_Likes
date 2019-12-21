import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import statsmodels.api as sm
import scipy.stats as stats


data = pd.read_csv(r"C:\Users\joeye\Downloads\STAT\university_data.csv", index_col = "University")



#%% 
#create scatter plot for one explanetory variable
X = data["Followers"]
X = sm.add_constant(X)
y = data["Likes"]

# Apply the regression equation

model = sm.OLS(y, X).fit()

# Determine the regression equation
b0 = round(model.params[0], 2)
b1 = round(model.params[1], 2)

print("The regression equation is: y = " + str(b0) + " + " + str(b1) + "x.")

sns.regplot(x="Followers", y="Likes", data=data, ci=0)
plt.title("Number of likes given number of Followers")
plt.ylabel("Number of Likes")
plt.xlabel("Number of Followers")
plt.xlim(0, 410000)
plt.ylim(0, 12000)
plt.show()

#%%
X = data[["Followers", "Comments", "Hashtags"]]
X = sm.add_constant(X)
y = data["Likes"]

# Apply the regression equation
model = sm.OLS(y, X).fit()

bj = round(model.params,3)
print(bj)

# Predicted value
yhat = model.fittedvalues[0]
yhat = round(yhat,1)
print(yhat)



#%% 
#number 2

# Store the residuals and predicted values
m_resid = model.resid
m_pred = model.fittedvalues

#%%
# Create residual plot 
sns.residplot(x=m_pred, y=m_resid)
plt.title("Residual plot (predicted)")
plt.ylabel("Residuals")
plt.xlabel("Predicted values")
plt.ylim(-10000, 10000)
plt.xlim(0, 9000)
#%%
# Create residual plot 
sns.residplot(x=data.Followers, y=m_resid)
plt.title("Residual plot (number of followers)")
plt.ylabel("Residuals")
plt.xlabel("Followers")
plt.ylim(-10000, 10000)
plt.xlim(0, 600000)

#%%
# Create residual plot 
sns.residplot(x=data.Comments, y=m_resid)
plt.title("Residual plot (number of Comments)")
plt.ylabel("Residuals")
plt.xlabel("Comments")
plt.ylim(-10000, 10000)
plt.xlim(0, 10000)

#%%
# Create residual plot 
sns.residplot(x=data.Hashtags, y=m_resid)
plt.title("Residual plot (hashtags)")
plt.ylabel("Residuals")
plt.xlabel("Hashtags")
plt.ylim(-10000, 10000)
plt.xlim(0, 10000)




#%%

model_output = model.summary()
#print(model_output)




#%%
X = data[["Followers"]]
X = sm.add_constant(X)
y = data["Likes"]

# Apply the regression equation
model = sm.OLS(y, X).fit()

model_output = model.summary()
print(model_output)
#%%

X_ext = data[["Followers", "Hashtags", "Comments"]]
X_ext = sm.add_constant(X_ext)
y = data["Likes"]

model_ext = sm.OLS(y, X_ext).fit()


r2_2 = model.rsquared
r2_1 = model_ext.rsquared

n = model.nobs
# Apply the regression equation

test_stat = ( (n-2) / 1 )*( (r2_1-r2_2) / (1-r2_1) )
test_stat = round(test_stat, 2)
print(test_stat)

#%%
# p-value

q = 2


pval = 1 - stats.f.cdf(test_stat, q, n-p-1)
pval = round(pval, 4)
print(pval)


#%%
# Partial F test
r2_2 = model.rsquared
r2_1 = model_ext.rsquared
 
p = 3
q = 2
n = model.nobs

# Test statistic
test_stat = ( (n-p-1) / q )*( (r2_1-r2_2) / (1-r2_1) )
test_stat = round(test_stat, 2)
print(test_stat)

#%%
# p-value
pval = 1 - stats.f.cdf(test_stat, q, n-p-1)
pval = round(pval, 4)
print(pval)


#%%
# Adjusted R-squared
print(model_output)

#%%
# Adjusted R-squared: original
adj_r2 = model.rsquared_adj
adj_r2 = round(adj_r2, 3)
print("ADJUSTED ORIG: ",adj_r2)

#%%
# Adjusted R-squared: extended
adj_r2_ext = model_ext.rsquared_adj
adj_r2_ext = round(adj_r2_ext, 3)
print("ADJUSTED EXTENDED: ", adj_r2_ext)