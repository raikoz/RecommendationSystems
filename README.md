# recommendation_systems
System that suggests movies a user would like in 40 lines of Python using the LightFM recommendation library

## Overview

The code uses the lightfm recommender system library to train a hybrid content-based + collaborative algorithm that uses the WARP loss function on the movielens dataset. The movielens dataset contains movies and ratings from over 1700 users. Once trained, the script prints out recommended movies for whatever users from the dataset that we choose to terminal.

## Dependencies

numpy (http://www.numpy.org/)
scipy (https://www.scipy.org/)
lightfm (https://github.com/lyst/lightfm)
Install missing dependencies using pip. lol yeah.

## Usage

Once you have your dependencies installed via pip, run the script in terminal via
```
python demo.py
```
Note If the lightfm dependency doesn't work for you via pip, just install it from source by running these two commands.
```
git clone git@github.com:lyst/lightfm.git
```
```
cd lightfm && pip install -e .
```
If you still have dependency version issues, use virtualenv.

## Credits

Credit goes to the [lightfm](https://github.com/lyst/lightfm) team and [siraj](https://www.youtube.com/channel/UCWN3xxRkmTPmbKwht9FuE5A). I've made it readable. Lmao.
