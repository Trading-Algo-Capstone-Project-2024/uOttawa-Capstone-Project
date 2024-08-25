original = df[['Date', 'Open']]
print(original)
original['Date'] = pd.to_datetime(original.loc[:, 'Date'])