"""
SUMMARY
================================================================================

These files contain 1000209 anonymous ratingDataset of approximately 3, 900 movies made by 6040 MovieLens users who
joined MovieLens in 2000.

USAGE LICENSE
================================================================================

Neither the University of Minnesota nor any of the researchers
involved can guarantee the correctness of the data, its suitability
for any particular purpose, or the validity of results based on the
use of the data set.  The data set may be used for any research
purposes under the following conditions:
 * The user may not state or imply any endorsement from the
       University of Minnesota or the GroupLens Research Group.

     * The user must acknowledge the use of the data set in
       publications resulting from the use of the data set
       (see below for citation information).

     * The user may not redistribute the data without separate
       permission.

     * The user may not use this information for any commercial or
       revenue-bearing purposes without first obtaining permission
       from a faculty member of the GroupLens Research Project at the
       University of Minnesota.

If you have any further questions or comments, please contact GroupLens
<grouplens-info@cs.umn.edu>.

CITATION
================================================================================
To acknowledge use of the dataset in publications, please cite the following
paper:

F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History
and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4,
Article 19 (December 2015), 19 pages. DOI=http://dx.doi.org/10.1145/2827872

ACKNOWLEDGEMENTS
================================================================================
Thanks to Shyong Lam and Jon Herlocker for cleaning up and generating the data
set.

FURTHER INFORMATION ABOUT THE GROUPLENS RESEARCH PROJECT
================================================================================

The GroupLens Research Project is a research group in the Department of
Computer Science and Engineering at the University of Minnesota. Members of
the GroupLens Research Project are involved in many research projects related
to the fields of information filtering, collaborative filtering, and
recommendation systems. The project is lead by professors John Riedl and Joseph
Konstan. The project began to explore automated collaborative filtering in
1992, but is most well known for its world wide trial of an automated
collaborative filtering system for Usenet news in 1996. Since then the project
has expanded its scope to research overall information filtering solutions,
integrating in content-based methods as well as improving current collaborative
filtering technology.

Further information on the GroupLens Research project, including research
publications, can be found at the following web site:

        http://www.grouplens.org/

GroupLens Research currently operates a movie recommender based on
collaborative filtering:

        http://www.movielens.org/

RATINGS FILE DESCRIPTION
================================================================================

All ratingDataset are contained in the file "ratingDataset.dat" and are in the
following format:

UserID::MovieID::Rating::Timestamp

- UserIDs range between 1 and 6040
- MovieIDs range between 1 and 3952
- Ratings are made on a 5-star scale (whole-star ratingDataset only)
- Timestamp is represented in seconds since the epoch as returned by time(2)
- Each user has at least 20 ratingDataset

USERS FILE DESCRIPTION
================================================================================

User information is in the file "users.dat" and is in the following
format:

UserID::Gender::Age::Occupation::Zip-code

All demographic information is provided voluntarily by the users and is
not checked for accuracy.  Only users who have provided some demographic
information are included in this data set.

- Gender is denoted by a "M" for male and "F" for female
- Age is chosen from the following ranges:

	*  1:  "Under 18"
	* 18:  "18-24"
	* 25:  "25-34"
	* 35:  "35-44"
	* 45:  "45-49"
	* 50:  "50-55"
	* 56:  "56+"

- Occupation is chosen from the following choices:

	*  0:  "other" or not specified
	*  1:  "academic/educator"
	*  2:  "artist"
	*  3:  "clerical/admin"
	*  4:  "college/grad student"
	*  5:  "customer service"
	*  6:  "doctor/health care"
	*  7:  "executive/managerial"
	*  8:  "farmer"
	*  9:  "homemaker"
	* 10:  "K-12 student"
	* 11:  "lawyer"
	* 12:  "programmer"
	* 13:  "retired"
	* 14:  "sales/marketing"
	* 15:  "scientist"
	* 16:  "self-employed"
	* 17:  "technician/engineer"
	* 18:  "tradesman/craftsman"
	* 19:  "unemployed"
	* 20:  "writer"

MOVIES FILE DESCRIPTION
================================================================================

Movie information is in the file "movies.dat" and is in the following
format:

MovieID::Title::Genres

- Titles are identical to titles provided by the IMDB (including
year of release)
- Genres are pipe-separated and are selected from the following genres:

	* Action
	* Adventure
	* Animation
	* Children's
	* Comedy
	* Crime
	* Documentary
	* Drama
	* Fantasy
	* Film-Noir
	* Horror
	* Musical
	* Mystery
	* Romance
	* Sci-Fi
	* Thriller
	* War
	* Western

- Some MovieIDs do not correspond to a movie due to accidental duplicate
entries and/or test entries
- Movies are mostly entered by hand, so errors and inconsistencies may exist
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import re

def find_all(name, path):
    result = []
    for root, dirs, files in os.walk(path):
        for filename in files:
            if re.search(name, filename) != None:
                result.append(os.path.join(root, filename))
    return result

# importing dataset
try:
    movienames = ['MovieID', 'Title', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary'
        , 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War'
        , 'Western']
    movieDataset = pd.read_csv('ml-1m/processed_movies.dat', '|', header = None, names = movienames)
except FileNotFoundError:
    movieDataset = pd.read_csv('ml-1m/movies.dat', '::', header = None, names = ['MovieID', 'Title', 'Genres'])
    print('Pre-processing item dataset.')
    movieDataset['Action'] = np.zeros((movieDataset.shape[0], 1))
    movieDataset['Adventure'] = np.zeros((movieDataset.shape[0], 1))
    movieDataset['Animation'] = np.zeros((movieDataset.shape[0], 1))
    movieDataset['Children'] = np.zeros((movieDataset.shape[0], 1))
    movieDataset['Comedy'] = np.zeros((movieDataset.shape[0], 1))
    movieDataset['Crime'] = np.zeros((movieDataset.shape[0], 1))
    movieDataset['Documentary'] = np.zeros((movieDataset.shape[0], 1))
    movieDataset['Drama'] = np.zeros((movieDataset.shape[0], 1))
    movieDataset['Fantasy'] = np.zeros((movieDataset.shape[0], 1))
    movieDataset['Film-Noir'] = np.zeros((movieDataset.shape[0], 1))
    movieDataset['Horror'] = np.zeros((movieDataset.shape[0], 1))
    movieDataset['Musical'] = np.zeros((movieDataset.shape[0], 1))
    movieDataset['Mystery'] = np.zeros((movieDataset.shape[0], 1))
    movieDataset['Romance'] = np.zeros((movieDataset.shape[0], 1))
    movieDataset['Sci-Fi'] = np.zeros((movieDataset.shape[0], 1))
    movieDataset['Thriller'] = np.zeros((movieDataset.shape[0], 1))
    movieDataset['War'] = np.zeros((movieDataset.shape[0], 1))
    movieDataset['Western'] = np.zeros((movieDataset.shape[0], 1))
    # movieDataset['No-genre'] = np.zeros((movieDataset.shape[0], 1)) no default behaviors

    genres = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
              'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    
    for i in range(movieDataset.shape[0]):
        itemGenres = movieDataset.iloc[i, :].Genres.split('|')
        for itemGenre in itemGenres:
            if itemGenre in genres:
                movieDataset.iloc[i, 3 + genres.index(itemGenre)] = 1
            elif itemGenre == 'Children\'s' or itemGenre == 'children\'s':
                movieDataset.iloc[i, 3 + genres.index('Children')] = 1
            else:
                # movieDataset.iloc[i, 21] = 1 no default behaviors
                continue

    to_new_dataset = list(range(21))
    to_new_dataset.remove(2)
    movieDataset = movieDataset.iloc[:, to_new_dataset]
    movieDataset.to_csv('ml-1m/processed_movies.dat', '|', index = False, header = False)
finally:
    try:
        usernames = ["UserId", "Gender", "Age", "Zipcode", "other", "academic", "artist", "clerical", "college",
                     "customer_service", "health care", "managerial", "farmer", "homemaker", "K-12 student", "lawyer",
                     "programmer", "retired", "sales", "scientist", "self-employed", "technician", "tradesman",
                     "unemployed", "writer"]
        userDataset = pd.read_csv('ml-1m/processed_users.dat', '|', header = None, names = usernames)
    except FileNotFoundError:
        userDataset = pd.read_csv('ml-1m/users.dat', '::', header = None,
                                  names = ['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'])
        print('Pre-processing user dataset.')
        keys = list(range(21))
        occupations = ["other", "academic", "artist", "clerical", "college", "customer_service", "health care",
                       "managerial", "farmer", "homemaker", "K-12 student", "lawyer", "programmer", "retired", "sales",
                       "scientist", "self-employed", "technician", "tradesman", "unemployed", "writer"]
        occupations = dict(zip(keys, occupations))
        for occupation in list(occupations.values()):
            userDataset[occupation] = np.zeros((userDataset.shape[0], 1))

        for i in range(userDataset.shape[0]):
            user = userDataset.iloc[i, :]
            if user.Gender == 'M':
                userDataset.iloc[i, 1] = 1
            elif user.Gender == 'F':
                userDataset.iloc[i, 1] = 0
            elif user.Gender == 1 or user.Gender == 0:
                print('Already encode genders')
            else:
                print('Corrupted or wrong input. Break!')
                break

            occupation = user.Occupation
            if occupation not in list(occupations):
                userDataset.iloc[i, 5] = 1
            else:
                userDataset.iloc[i, 5 + occupation] = 1

        to_new_dataset = list(range(userDataset.shape[1]))
        to_new_dataset.remove(3)
        userDataset = userDataset.iloc[:, to_new_dataset]
        userDataset.to_csv('ml-1m/processed_users.dat', '|', index = False, header = False)
    finally:
        ratingDataset = pd.read_csv('ml-1m/ratings.dat', '::', header = None,
                                    names = ['UserID', 'MovieID', 'Rating', 'Timestamp'])


users = userDataset.iloc[:, :].values
ratings = ratingDataset.iloc[:, [0, 1, 2]].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_users = LabelEncoder()
users[:, 3] = labelEncoder_users.fit_transform(users[:, 3])

oneHotEncoder = OneHotEncoder(categorical_features = [3])
users = oneHotEncoder.fit_transform(users).toarray()


#using the elbow method to select K
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 500, n_init = 20, random_state = 0)
    kmeans.fit(users)
    wcss.append(kmeans.inertia_)

f = plt.figure(1)
plt.plot(range(1, 11), wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')

print('We will choose number of cluster = 3.')

#K-means
kmeans = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 500, n_init = 20, random_state = 0)
y_kmeans = kmeans.fit_predict(users)

g1 = []
g2 = []
g3 = []

for i in range(len(y_kmeans)):
    if y_kmeans[i] == 0:
        g1.append(i)
    elif y_kmeans[i] == 1:
        g2.append(i)
    else:
        g3.append(i)

# users into 3 groups
groupOne = userDataset.iloc[g1, :].reset_index(drop = True)
groupTwo = userDataset.iloc[g2, :].reset_index(drop = True)
groupThree = userDataset.iloc[g3, :].reset_index(drop = True)

# statistic for each user groups
groupOne_stat = {'Gender': groupOne.Gender.mean(), 'Age': groupOne.Age.mean(),
                 "other": groupOne.other.sum(), "academic": groupOne.academic.sum(),
                 "artist": groupOne.artist.sum(), "clerical": groupOne.clerical.sum(),
                 "college": groupOne.college.sum(), "customer_service": groupOne.customer_service.sum(),
                 "health care": groupOne['health care'].sum(), "managerial": groupOne.managerial.sum(),
                 "farmer": groupOne.farmer.sum(),b"homemaker": groupOne.homemaker.sum(),
                 "K-12 student": groupOne['K-12 student'].sum(), "lawyer": groupOne.lawyer.sum(),
                 "programmer": groupOne.programmer.sum(), "retired": groupOne.retired.sum(),
                 "sales": groupOne.sales.sum(), "scientist": groupOne.scientist.sum(),
                 "self-employed": groupOne['self-employed'].sum(), "technician": groupOne.technician.sum(),
                 "tradesman": groupOne.tradesman.sum(), "unemployed": groupOne.unemployed.sum(),
                 "writer": groupOne.writer.sum(), 'No_user': groupOne.shape[0]}

groupTwo_stat = {'Gender': groupTwo.Gender.mean(), 'Age': groupTwo.Age.mean(),
                 "other": groupTwo.other.sum(), "academic": groupTwo.academic.sum(),
                 "artist": groupTwo.artist.sum(), "clerical": groupTwo.clerical.sum(),
                 "college": groupTwo.college.sum(), "customer_service": groupTwo.customer_service.sum(),
                 "health care": groupTwo['health care'].sum(), "managerial": groupTwo.managerial.sum(),
                 "farmer": groupTwo.farmer.sum(),b"homemaker": groupTwo.homemaker.sum(),
                 "K-12 student": groupTwo['K-12 student'].sum(), "lawyer": groupTwo.lawyer.sum(),
                 "programmer": groupTwo.programmer.sum(), "retired": groupTwo.retired.sum(),
                 "sales": groupTwo.sales.sum(), "scientist": groupTwo.scientist.sum(),
                 "self-employed": groupTwo['self-employed'].sum(), "technician": groupTwo.technician.sum(),
                 "tradesman": groupTwo.tradesman.sum(), "unemployed": groupTwo.unemployed.sum(),
                 "writer": groupTwo.writer.sum(), 'No_user': groupTwo.shape[0]}

groupThree_stat = {'Gender': groupThree.Gender.mean(), 'Age': groupThree.Age.mean(),
                 "other": groupThree.other.sum(), "academic": groupThree.academic.sum(),
                 "artist": groupThree.artist.sum(), "clerical": groupThree.clerical.sum(),
                 "college": groupThree.college.sum(), "customer_service": groupThree.customer_service.sum(),
                 "health care": groupThree['health care'].sum(), "managerial": groupThree.managerial.sum(),
                 "farmer": groupThree.farmer.sum(), "homemaker": groupThree.homemaker.sum(),
                 "K-12 student": groupThree['K-12 student'].sum(), "lawyer": groupThree.lawyer.sum(),
                 "programmer": groupThree.programmer.sum(), "retired": groupThree.retired.sum(),
                 "sales": groupThree.sales.sum(), "scientist": groupThree.scientist.sum(),
                 "self-employed": groupThree['self-employed'].sum(), "technician": groupThree.technician.sum(),
                 "tradesman": groupThree.tradesman.sum(), "unemployed": groupThree.unemployed.sum(),
                 "writer": groupThree.writer.sum(), 'No_user': groupThree.shape[0]}

g1_stat = 'Avarage male-female split, older, possibly make the most money'
g2_stat = 'More male, youngest, has a lot of students'
g3_stat = 'More female than average, median age group, medium income'

print()
print('___________________________________________________')
print('Brief summary of group 1: {}'.format(g1_stat))
print('Brief summary of group 2: {}'.format(g2_stat))
print('Brief summary of group 3: {}'.format(g3_stat))


g1_id = groupOne.UserId.tolist()
g2_id = groupTwo.UserId.tolist()
g3_id = groupThree.UserId.tolist()

number_people_in_g1 = len(g1_id)
number_people_in_g2 = len(g2_id)
number_people_in_g3 = len(g3_id)

g1_rating = ratingDataset[ratingDataset.UserID.isin(g1_id)].reset_index(drop = True)
g2_rating = ratingDataset[ratingDataset.UserID.isin(g2_id)].reset_index(drop = True)
g3_rating = ratingDataset[ratingDataset.UserID.isin(g3_id)].reset_index(drop = True)

g1_rating_stat = pd.DataFrame({'sum': g1_rating.groupby('MovieID').Rating.sum(),
                               'number_of_ratings': g1_rating.groupby('MovieID').Rating.count(),
                               'mean': g1_rating.groupby('MovieID').Rating.mean(), 
                               'stdev': g1_rating.groupby('MovieID').Rating.std(), 
                               'var': g1_rating.groupby('MovieID').Rating.var()})

g2_rating_stat = pd.DataFrame({'sum': g2_rating.groupby('MovieID').Rating.sum(),
                               'number_of_ratings': g2_rating.groupby('MovieID').Rating.count(),
                               'mean': g2_rating.groupby('MovieID').Rating.mean(), 
                               'stdev': g2_rating.groupby('MovieID').Rating.std(), 
                               'var': g2_rating.groupby('MovieID').Rating.var()})

g3_rating_stat = pd.DataFrame({'sum': g3_rating.groupby('MovieID').Rating.sum(),
                               'number_of_ratings': g3_rating.groupby('MovieID').Rating.count(),
                               'mean': g3_rating.groupby('MovieID').Rating.mean(), 
                               'stdev': g3_rating.groupby('MovieID').Rating.std(), 
                               'var': g3_rating.groupby('MovieID').Rating.var()})

g1_rating_stat.reset_index(inplace = True)
g2_rating_stat.reset_index(inplace = True)
g3_rating_stat.reset_index(inplace = True)

'''
g1_rating_stat.to_csv('ml-1m/g1_rating_stat.dat', ',', index = None)
g2_rating_stat.to_csv('ml-1m/g2_rating_stat.dat', ',', index = None)
g3_rating_stat.to_csv('ml-1m/g3_rating_stat.dat', ',', index = None)
'''

print('Files was already created and ready to use. Three lines to write out to file will be comment. If you really '
      'want to rewrite the file, copy the if part to console to run it.')

# take movie that has mean(rating) >= 4, standard deviation <= 0.8 and had been rated by
# at least 10 percent of the group, sorted by mean desc, stdev asc
g1_rating_stat = g1_rating_stat.loc[g1_rating_stat.number_of_ratings >= 0.1 * number_people_in_g1]
g2_rating_stat = g2_rating_stat.loc[g2_rating_stat.number_of_ratings >= 0.1 * number_people_in_g2]
g3_rating_stat = g3_rating_stat.loc[g3_rating_stat.number_of_ratings >= 0.1 * number_people_in_g3]

g1_rating_stat = g1_rating_stat.loc[g1_rating_stat['mean'] >= 4]
g2_rating_stat = g2_rating_stat.loc[g2_rating_stat['mean'] >= 4]
g3_rating_stat = g3_rating_stat.loc[g3_rating_stat['mean'] >= 4]

g1_rating_stat = g1_rating_stat.loc[g1_rating_stat['stdev'] <= 0.8]
g2_rating_stat = g2_rating_stat.loc[g2_rating_stat['stdev'] <= 0.8]
g3_rating_stat = g3_rating_stat.loc[g3_rating_stat['stdev'] <= 0.8]

g1_rating_stat.sort_values(['mean', 'stdev'], 0, [False, True], True)
g2_rating_stat.sort_values(['mean', 'stdev'], 0, [False, True], True)
g3_rating_stat.sort_values(['mean', 'stdev'], 0, [False, True], True)

g1_rating_stat.reset_index(drop = True, inplace = True)
g2_rating_stat.reset_index(drop = True, inplace = True)
g3_rating_stat.reset_index(drop = True, inplace = True)

# list of 60 highest ranking movies in each group by user
fav_movies_g1 = g1_rating_stat.iloc[0: 61, 0].values.tolist()
fav_movies_g2 = g2_rating_stat.iloc[0: 61, 0].values.tolist()
fav_movies_g3 = g3_rating_stat.iloc[0: 61, 0].values.tolist()

by_genres_g1 = {'Action': 0, 'Adventure': 0, 'Animation': 0, 'Children': 0, 'Comedy': 0, 'Crime': 0, 'Documentary': 0,
                'Drama': 0, 'Fantasy': 0, 'Film-Noir': 0, 'Horror': 0, 'Musical': 0, 'Mystery': 0, 'Romance': 0,
                'Sci-Fi': 0, 'Thriller': 0, 'War': 0, 'Western': 0}
by_genres_g2 = {'Action': 0, 'Adventure': 0, 'Animation': 0, 'Children': 0, 'Comedy': 0, 'Crime': 0, 'Documentary': 0,
                'Drama': 0, 'Fantasy': 0, 'Film-Noir': 0, 'Horror': 0, 'Musical': 0, 'Mystery': 0, 'Romance': 0,
                'Sci-Fi': 0, 'Thriller': 0, 'War': 0, 'Western': 0}
by_genres_g3 = {'Action': 0, 'Adventure': 0, 'Animation': 0, 'Children': 0, 'Comedy': 0, 'Crime': 0, 'Documentary': 0,
                'Drama': 0, 'Fantasy': 0, 'Film-Noir': 0, 'Horror': 0, 'Musical': 0, 'Mystery': 0, 'Romance': 0,
                'Sci-Fi': 0, 'Thriller': 0, 'War': 0, 'Western': 0}

for movie in fav_movies_g1:
    genres = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
              'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    line = movieDataset.loc[movieDataset.MovieID == movie]
    line = line.iloc[:, 2:].values.tolist()[0]
    tmp = dict(zip(genres, line))
    for genre in list(tmp):
        by_genres_g1[genre] += tmp[genre]

for movie in fav_movies_g2:
    genres = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
              'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    line = movieDataset.loc[movieDataset.MovieID == movie]
    line = line.iloc[:, 2:].values.tolist()[0]
    tmp = dict(zip(genres, line))
    for genre in list(tmp):
        by_genres_g2[genre] += tmp[genre]

for movie in fav_movies_g3:
    genres = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
              'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    line = movieDataset.loc[movieDataset.MovieID == movie]
    line = line.iloc[:, 2:].values.tolist()[0]
    tmp = dict(zip(genres, line))
    for genre in list(tmp):
        by_genres_g3[genre] += tmp[genre]

print('\nGroup 1')
for item in by_genres_g1:
    print(item + ': ' + str(int(by_genres_g1[item])))

print('\nGroup 2')
for item in by_genres_g2:
    print(item + ': ' + str(int(by_genres_g2[item])))

print('\nGroup 3')
for item in by_genres_g3:
    print(item + ': ' + str(int(by_genres_g3[item])))

result = pd.DataFrame(by_genres_g1, index = [0])
result = result.append(pd.DataFrame(by_genres_g2, index = [0]), ignore_index = True)
result = result.append(pd.DataFrame(by_genres_g3, index = [0]), ignore_index = True)

result.to_csv('ml-1m/result.dat', ',', index = None)
