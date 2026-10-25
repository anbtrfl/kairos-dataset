# lab 1: predicting post performance

## 1. results

the final model is a single HistGradientBoostingClassifier with features from the text, author profile and previous posts. the code is in `model.py`.

final model, main validation set, seed 42: macro-F1 0.6515, QWK 0.5807, accuracy 0.6943.
final model, official test set: macro-F1 0.6439, QWK 0.5759, accuracy 0.6872.

the original baseline was reproduced with a test macro-F1 of 0.5410. the final model improved this score by 0.1029.

## 2. data split

the official training set contains 71,395 labelled posts from 3,295 authors

all main experiments used the same validation split with seed 42.
TF-IDF, SVD, scaling and category encoding were fitted only on the training part of each experiment. 

to check whether the result depended on the seed 42 split, the earlier model with photos was also trained and evaluated on another split with seed 123. the features and model settings stayed the same, while TF-IDF and SVD were fitted again on the new training part. macro-F1 was 0.6574, compared with 0.6524 on seed 42. this supports that version's stability across the two splits, although the validation sets partly overlap. the final model was checked on validation seed 42 and then trained on all 71,395 official training posts for evaluation on the 17,623 test posts.

after the first test result was seen, further experiments used validation. the final model was selected based on validation performance and simplicity. the final version was then evaluated on the same official test set, which had already been inspected.

## 3. solution history

experiments used 56,847 training posts from 2,636 authors and 14,548 validation posts from 659 authors. each new feature was compared with the model it was added to.

### first experiments

**baseline.** first, the original baseline was reproduced: 0.5410 on the official test and 0.5546 on our validation. it uses information such as text length, links, hashtags, posting time, followers and previous reactions. this gave a starting point for checking whether new features helped.

**text only.** logistic regression with TF-IDF gave 0.4422. TF-IDF represents a post through its words and short phrases. text alone was weaker than the baseline, which suggested that the author's usual performance was important for this task.

**combining text and baseline features.** logistic regression on baseline features gave 0.5115. adding TF-IDF increased this to 0.5631. text was useful together with author information, even though it was weak on its own.

**SVD and boosting.** the large TF-IDF matrix was reduced to 100 SVD components and combined with baseline features in one boosting model. this gave 0.5821. SVD made the text representation smaller and suitable for combining with numerical features.

### history and profile

**similarity to earlier posts.** two features described similarity to the most similar earlier post and to the average earlier text. they gave 0.5833. adding the mean resid of the three most similar earlier posts increased the result to 0.5985. the idea was to use how similar content had performed for the same author.

**profile.** simple profile features gave 0.5988. replacing them with profile text represented by TF-IDF and 50 SVD components gave 0.6007. the text included the headline, summary, skill names and job titles.

**class weights.** a version without weights gave 0.5968. moderate weights, calculated as the square root of balanced weights, gave 0.6281. this helped the model pay more attention to below and above without fully balancing the classes.

**recent history.** the last previous resid and the mean resid of the last five posts increased the score to 0.6323. these features described the author's recent results as well as their longer history.

**follower history.** i would also like to have follower counts over time, rather than only the count at collection. changes in audience size could affect reactions to later posts. recent reaction history may capture some of these changes indirectly, but it cannot tell us whether they came from changes in followers or something else. follower history available before each post might therefore help, but this could not be tested with the available data.

### text embeddings and media

**full-post E5.** adding frozen E5 vectors for the full text to the 0.6323 model gave 0.6362. adding post-to-profile similarity in the related E5 version gave 0.6342. the neural network was used to create features, not fine-tuned on our labels.

**photos on the earlier model.** photo counts gave 0.6321, while CLIP photo vectors with counts gave 0.6351, compared with 0.6323 without photos. on the 2,521 validation posts with photos, the scores were 0.5812 without photo features, 0.5862 with counts and 0.5801 with CLIP and counts. the benefit of photo content was therefore not clear at this stage.

**video.** CLIP vectors from frames at 10%, 50% and 90% of video duration gave 0.6279 on the 0.6323 model. video features were not kept.

**repost flags.** adding the reposted flag and the presence of a nested reshared post to the 0.6323 model gave 0.6344. a nested post was not automatically treated as a repost without a comment.

**post opening E5.** the first nonempty line, up to 300 characters, was encoded with multilingual-e5-small. together with repost flags, this gave 0.6415. adding full-post E5 vectors to this version gave 0.6412, and replacing history similarity with E5 similarity gave 0.6415. neither extra change was kept.

### later additions and final simplification

**relative author style.** text length, content type, links and hashtags were compared with the author's earlier posts. for example, a long post may be normal for one author but unusual for another. these four features improved 0.6415 to 0.6455.

**photos on the stronger model.** CLIP photo vectors and photo counts improved 0.6455 to 0.6474 and were kept at this stage. this experiment did not separate the effects of CLIP content and counts.

**reshared text.** a review of JSON fields showed that the text inside resharedPost was not yet used. a separate TF-IDF representation with 50 SVD components improved 0.6474 to 0.6510.

**history of the same content type.** two features were added: the number of earlier posts of the current type and their median log reaction level relative to the author's current level. at least three earlier posts were required for the median feature. this improved 0.6510 to 0.6524 in the version with photos. these two history features are kept in the final model.

**removing photos.** removing all 512 CLIP coordinates and four photo flags/counts reduced macro-F1 only from 0.6524 to 0.6515. QWK was 0.5807 and accuracy was 0.6943. i chose this simpler version: it needs 615 numerical features instead of 1,131 and does not require CLIP or photo processing.

## 4. other experiments (not lucky)

these changes were not kept. scores are validation macro-F1; each result is compared with its own base model:

- extra recent-history statistics: 0.6476 vs 0.6474, too small a gain;
- changes in the author's recent reaction level: 0.6489 vs 0.6510;
- article/entity titles and descriptions: 0.6509 vs 0.6510;
- profile country, language, industry and company size: 0.6492 vs 0.6510;
- earlier mentions and reshared-source features: 0.6506 vs 0.6510;
- full-text SVD with 250 components: 0.6498 vs 0.6510;
- 300 boosting iterations without early stopping: 0.6502 vs 0.6510;
- posting rhythm: 0.6505 vs 0.6524;
- min_samples_leaf=50: 0.6497 vs 0.6524;
- class weight exponent 0.7: 0.6493 vs 0.6524;
- separate history by repost status: 0.6480 vs 0.6524;
- normalized PERSON codes: 0.6512 vs 0.6524;
- two binary classifiers for the ordered classes: 0.6456 vs 0.6524.

text logistic regression probabilities improved 0.6524 to 0.6535, but the extra training steps were not worth the small gain for this project. adding CLIP text-photo similarity gave 0.6531. smaller boosting steps of 0.04 and 0.06 gave 0.6531 and 0.6530, compared with 0.6535. these more complex versions were not evaluated on test.

## 5. classifier comparison

three classifiers were compared on the same features of the 0.6510 version, before adding history of the same content type. boosting gave macro-F1 0.6510, QWK 0.5807 and accuracy 0.6947. logistic regression gave 0.6332, 0.5607 and 0.6712. random forest gave 0.5751, 0.4713 and 0.6888. boosting was selected by macro-F1.

logistic regression used median imputation, scaling, C=1 and max_iter=2000. random forest used median imputation, 300 trees, min_samples_leaf=5 and max_features="sqrt". all models used moderate class weights and were trained one after another. this was a comparison of fixed settings, not a full search for each algorithm.

## 6. final model

the model combines text, profile and author history features in one HistGradientBoostingClassifier.

- post text: TF-IDF → SVD, 100 components
- profile and reshared text: separate TF-IDF -> SVD, 50 components each
- post opening: 384 frozen E5 features from the first nonempty line, up to 300 characters
- other features: baseline features, repost flags, relative writing style and results of earlier posts, including posts of the same type.

boosting uses learning_rate=0.08, max_iter=300, min_samples_leaf=20 and random_state=42. most settings were kept from the baseline. early stopping is enabled, so training can finish before 300 iterations. class weights are `(len(train) / (3 * class_count)) ** 0.5` to give more attention to the less frequent classes

## 7. errors

below was the main difficulty. on the final test, its precision was 0.5443, recall 0.5140 and F1 0.5287. 47.33% of true below posts were predicted as typical. stronger class weights and two binary classifiers helped find more below posts during validation experiments, but caused too many false predictions for typical and reduced overall macro-F1. separate history by repost status did not help either. these changes were not kept.

## 8. files

- `experiments/`: earlier model versions and classifier comparisons.
- `model.py`: the final model.
- `train.py`: run with `python train.py`. the calls at the end select validation or training on the full official training set with test evaluation. the final test call is currently enabled.
- `prepare*.py`, `profile_prepare.py` and `aggregate_media.py`: data preparation and cached text and media vectors.
- the original dataset is stored on an external drive, outside the repository. its paths are set in the scripts.
