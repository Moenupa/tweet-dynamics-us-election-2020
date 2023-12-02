import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# bidenData = pd.read_csv('data/src/hashtag_joebiden.csv', lineterminator='\n')
# bidenEmot = pd.read_csv('data/emotion/biden.csv')
# bidenSent = pd.read_csv('data/sent/biden.csv')
bidenData = pd.read_csv('data/src/hashtag_donaldtrump.csv', lineterminator='\n')
bidenEmot = pd.read_csv('data/emotion/trump.csv')
bidenSent = pd.read_csv('data/sent/trump.csv')
len = len(bidenData)
date = []
positive = dict()
negative = dict()
neutral = dict()
count = dict()
for i in range(len):
    dateStr = bidenData['collected_at'][i]
    if not isinstance(dateStr, str):
        # print("err! {2:}th dateStr is not string, type = {0:}, val = {1:}".format(type(dateStr), dateStr, i))
        continue
    if '.' in dateStr:
        pre, post = dateStr.split('.')
        dateStr = pre + '.' + post[:6]
        date.append(datetime.strptime(dateStr, '%Y-%m-%d %H:%M:%S.%f'))
    else:
        date.append(datetime.strptime(dateStr, '%Y-%m-%d %H:%M:%S'))
    simpDate = datetime(date[-1].year, date[-1].month, date[-1].day, date[-1].hour)
    count[simpDate] = count.get(simpDate, 0) + 1
    if bidenSent['sent'][i] == 'positive':
        positive[simpDate] = positive.get(simpDate, 0) + 1
    elif bidenSent['sent'][i] == 'negative':
        negative[simpDate] = negative.get(simpDate, 0) + 1
    else:
        neutral[simpDate] = neutral.get(simpDate, 0) + 1

sum = 0
for date, val in count.items():
    print('{0:} : {1:}'.format(date, val))
    sum += val

x = []
ypos = []
yneu = []
yneg = []
for k in sorted(count.keys()):
    x.append(k)
    ypos.append(positive[k] / count[k] * 100)
    yneu.append(neutral[k] / count[k] * 100)
    yneg.append(negative[k] / count[k] * 100)

plt.stackplot(x, yneg, yneu, ypos, labels = ['negative', 'neutral', 'positive'], colors = ['#F7756D', '#6DEFF7', '#6DF798'])
plt.plot(x, yneg, color='#E6645C', linestyle='-')
plt.plot(x, np.array(yneg) + np.array(yneu), color='#5CE687', linestyle='-')
plt.xlabel('time(MM-DD)')
plt.ylabel('rate(%)')
plt.xlim(x[0], x[-1])
plt.ylim(0, 100)
plt.title('sentiment rate in #Trump')
plt.legend(loc = 'lower right')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
plt.gcf().autofmt_xdate()
plt.savefig("figures/figure2.png")
plt.show()
