import numpy as np
import pandas as pd
import pickle
import math


def main():
    ratings_header = ['user_id', 'item_id', 'rating', 'timestamp']
    ratings = pd.read_csv('data/ml-100k/u.data', sep='\t', names=ratings_header)

    n_ratings = len(ratings)
    n_users = ratings.user_id.unique().shape[0]
    n_items = ratings.item_id.unique().shape[0]
    print('Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_items))

    # Users from preprocessed user file with locations
    # ['user_id', 'age', 'sex', 'occupation', 'zip_code', 'loc_lon', 'loc_lat']

    users = pickle.load(open('data/users.pickle', 'rb'))

    n_features_user = 4  # Age, Sex, Location (lon, lat)
    n_occupations = users.occupation.unique().shape[0]
    n_features_user += n_occupations   # HotOne encoded

    occupation = pd.get_dummies(users.iloc[:, 3])

    # Item Data
    items_header = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url', 'unknown|0', 'Action|1', 'Adventure|2', 'Animation|3', 'Children\'s|4', 'Comedy|5', 'Crime|6', 'Documentary|7', 'Drama|8', 'Fantasy|9', 'Film-Noir|10', 'Horror|11', 'Musical|12', 'Mystery|13', 'Romance|14', 'Sci-Fi|15', 'Thriller|16', 'War|17', 'Western|18']
    items = pd.read_csv('data/ml-100k/u.item', sep='|', names=items_header, encoding='latin-1')

    n_features_item = 19 + 2  # Genre Tags

    meta_users = np.zeros((n_users, n_features_user))
    meta_items = np.zeros((n_items, n_features_item))

    # Fill user metadate matrix
    meta_users[:, 0] = users.iloc[:, 1]  # Age
    meta_users[:, 1] = [-1 if x == 'M' else 1 for x in users.iloc[:, 2]]  # Sex
    meta_users[:, 2:4] = users.iloc[:, [5, 6]]  # Location (lon, lat)
    meta_users[:, 4:4+n_occupations] = occupation.iloc[:, :]

    # Fill movie metadata matrix
    meta_items[:, 0:19] = items.iloc[:, 5:]

    import imdbpie
    imdb = imdbpie.Imdb()

    # for item in items.itertuples():
    iterator_items = items.itertuples()

    while True:
        item = next(iterator_items)
        print(item[1])
        title = item[2]
        sp = title.split('(')
        titles, year = sp[:-1], sp[-1]

        titles_formated = []
        for title in titles:
            title = title.strip(' )')
            if ',' in title:
                sp = title.split(',')
                sp = [x.strip(' ') for x in sp]
                if len(sp) == 2:
                    if len(sp[1]) < 4:
                        title = '{} {}'.format(sp[1], sp[0])
            titles_formated.append(title)

        year = int(year.replace(')', ''))

        found = imdb.search_for_title(titles_formated[0])

        def checktitles(titles, lookup):
            lookup = lookup.lower().replace(',', '').replace('&', 'and')
            for title in titles:
                title = title.lower().replace(',', '').replace('&', 'and')
                if lookup.find(title) != -1 or title.find(lookup) != -1:
                    return True

            return False

        id_is = -1
        for i, f in enumerate(found):
            if checktitles(titles_formated, found[i]['title']) and found[i]['year'] == str(year):
                id_is = i
                break

        if id_is == -1:
            for i, f in enumerate(found):
                if checktitles(titles_formated, found[i]['title']) and abs(int(found[i]['year']) - year) < 2:
                    id_is = i
                    break

        if id_is != -1:
            movie = imdb.get_title_by_id(found[id_is]['imdb_id'])
            meta_items[item[0] - 1, 19] = movie.rating
            meta_items[item[0] - 1, 20] = movie.year
        else:
            id_is = 0
            movie = imdb.get_title_by_id(found[id_is]['imdb_id'])
            meta_items[item[0] - 1, 19] = movie.rating
            meta_items[item[0] - 1, 20] = movie.year
            print('No good match, taking: {} ({}) --- {} ({})'.format(title, year, movie.title, movie.year))
            # print('Invalid: {} ({})'.format(title, year))
            # break

    movies_invalid = [267,]

    X = np.zeros((n_ratings, (n_features_user + n_features_item)))
    U = np.zeros(n_ratings)
    I = np.zeros(n_ratings)
    Y = np.zeros(n_ratings)

    n_invalid = 0

    id = 0

    # ['Age', 'Sex', 'Loc_Lon', 'Loc_Lat', 'job_is_administrator', 'job_is_artist', 'job_is_doctor', 'job_is_educator', 'job_is_engineer', 'job_is_entertainment', 'job_is_executive', 'job_is_healthcare', 'job_is_homemaker', 'job_is_lawyer', 'job_is_librarian', 'job_is_marketing', 'job_is_none', 'job_is_other', 'job_is_programmer', 'job_is_retired', 'job_is_salesman', 'job_is_scientist', 'job_is_student', 'job_is_technician', 'job_is_writer', 'unknown|0', 'Action|1', 'Adventure|2', 'Animation|3', 'Children\'s|4', 'Comedy|5', 'Crime|6', 'Documentary|7', 'Drama|8', 'Fantasy|9', 'Film-Noir|10', 'Horror|11', 'Musical|12', 'Mystery|13', 'Romance|14', 'Sci-Fi|15', 'Thriller|16', 'War|17', 'Western|18']

    for rating in ratings.itertuples():
        # id = rating.Index
        userid = rating[1]
        itemid = rating[2]
        rating_score = rating[3]

        if math.isnan(users.iloc[userid - 1, 5]):
            n_invalid += 1
            continue

        U[id] = userid - 1
        I[id] = itemid - 1

        X[id, 0] = users.iloc[userid - 1, 1]  # Age
        X[id, 1] = -1 if users.iloc[userid - 1, 2] == 'M' else 1  # Sex
        X[id, 2:4] = users.iloc[userid - 1, [5, 6]]  # Location (lon, lat)
        X[id, 4:4+n_occupations] = occupation.iloc[userid - 1, :]
        X[id, 25:] = items.iloc[itemid - 1, 5:]

        Y[id] = rating_score

        id += 1

    X = X[:-n_invalid, :]
    Y = Y[:-n_invalid]
    U = U[:-n_invalid]
    I = I[:-n_invalid]

    print('{} invalid samples'.format(n_invalid))

    pickle.dump((X, U, I, Y), open('data/cont.pickle', 'wb'), pickle.HIGHEST_PROTOCOL)

    pickle.dump((meta_users, meta_items, U, I, Y), open('data/ratings_metadata.pickle', 'wb'), pickle.HIGHEST_PROTOCOL)

    # pickle.dump((meta_users, meta_items), open('data/imdb_metadata.pickle', 'wb'), pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
