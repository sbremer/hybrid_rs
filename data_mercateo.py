import pandas as pd
import numpy as np


header_c2bs = ['ProductType', 'Katalog', 'Lieferant', 'Anbieter', 'Kundennr', 'Firma', 'Land', 'Anrede', 'Vorname', '',
               'Nachname', 'Email', 'Tel', 'Firmengr', 'Branche', 'Vertriebsverantwortlicher', 'Status', 'Reg-Datum',
               '', 'Orders_ges', 'Umsatz_ges', 'Marge_ges', 'Last_Order ges', 'Subscription', 'Orders_seit_Subscript',
               'Umsatz_seit_Subscript', 'Marge_seit_Subscript', 'Orders_Katalog', 'Umsatz_Katalog', 'Transaktionsgeb',
               'Last_Order_Katalog', 'AffiliateId', 'AffiliateName', 'AffiliateDescr', '', '', '']
data = pd.read_csv('data/mercateo/CustomerToBusinessShop.csv', skiprows=5, names=header_c2bs, thousands=',')

users = data.Kundennr.unique()
items = data.Katalog.unique()

n_users = len(users)
n_items = len(items)

user_lookup = dict(zip(users, range(n_users)))
item_lookup = dict(zip(items, range(n_items)))

sales = np.zeros((n_users, n_items))
no_sale = 0
total_sales = 0.0

for row in data.itertuples():
    if row.Umsatz_Katalog > 0.0:
        user_id = user_lookup[row.Kundennr]
        item_id = item_lookup[row.Katalog]
        sale = row.Umsatz_Katalog
        sales[user_id, item_id] = sale
        total_sales += sale
    else:
        no_sale += 1

entries = len(sales.nonzero()[0])
sparsity = float(entries)
sparsity /= (sales.shape[0] * sales.shape[1])
print('Number of users {}'.format(n_users))
print('Number of Items {}'.format(n_items))
print('Total valid entries {}'.format(entries))
print('Entries with no sales volume {}'.format(no_sale))
print('Sparsity {:4.4f}%'.format(sparsity * 100))
print('Total sales volume {:,}'.format(int(total_sales)))
