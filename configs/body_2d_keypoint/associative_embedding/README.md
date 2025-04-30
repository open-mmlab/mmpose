# Associative embedding: End-to-end learning for joint detection and grouping (AE)

Associative Embedding is one of the most popular 2D bottom-up pose estimation approaches, that first detect all the keypoints and then group/associate them into person instances.

In order to group all the predicted keypoints to individuals, a tag is also predicted for each detected keypoint. Tags of the same person are similar, while tags of different people are different. Thus the keypoints can be grouped according to the tags.

<div align=center>
<img src="https://user-images.githubusercontent.com/15977946/146514181-84f22623-6b73-4656-89b8-9e7f551e9cc0.png">
</div>
