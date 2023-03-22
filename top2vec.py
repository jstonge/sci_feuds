import logging
from sklearn.cluster import dbscan
from sklearn.preprocessing import normalize

import hdbscan
import numpy as np
import pandas as pd
import umap

logger = logging.getLogger('top2vec')
logger.setLevel(logging.WARNING)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(sh)


class Top2Vec:
    def __init__(self,
                 document_embeddings,
                 document_ids,
                 abstracts,
                 umap_args=None,
                 hdbscan_args=None,
                 verbose=True
                 ):

        if verbose:
            logger.setLevel(logging.DEBUG)
            self.verbose = True
        else:
            logger.setLevel(logging.WARNING)
            self.verbose = False

        # validate document ids

        if not (isinstance(document_ids, list) or isinstance(document_ids, np.ndarray)):
            raise ValueError("Documents ids need to be a list of str or int")

        if len(document_ids) != len(set(document_ids)):
            raise ValueError("Document ids need to be unique")

        if all((isinstance(doc_id, str) or isinstance(doc_id, np.str_)) for doc_id in document_ids):
            self.doc_id_type = np.str_
        elif all((isinstance(doc_id, int) or isinstance(doc_id, np.int_)) for doc_id in document_ids):
            self.doc_id_type = np.int_
        else:
            raise ValueError("Document ids need to be str or int")

        self.document_ids_provided = True
        self.document_ids = np.array(document_ids)
        self.doc_id2index = dict(zip(document_ids, list(range(0, len(document_ids)))))

        self.document_vectors = document_embeddings

        self.abstracts = np.array(abstracts, dtype="object")

        # create 5D embeddings of documents
        logger.info('Creating lower dimension embedding of documents')

        if umap_args is None:
            umap_args = {'n_neighbors': 15,
                         'n_components': 5,
                         'metric': 'cosine'}

        umap_model = umap.UMAP(**umap_args).fit(self.document_vectors)

        if umap_args['n_components'] == 2:
            self.embedding = umap_model.embedding_

        # find dense areas of document vectors
        logger.info('Finding dense areas of documents')

        if hdbscan_args is None:
            hdbscan_args = {'min_cluster_size': 15,
                            'metric': 'euclidean',
                            'cluster_selection_method': 'eom'}

        cluster = hdbscan.HDBSCAN(**hdbscan_args).fit(umap_model.embedding_)

        # calculate topic vectors from dense areas of documents
        logger.info('Finding topics')

        # create topic vectors
        self._create_topic_vectors(cluster.labels_)

        # deduplicate topics
        self._deduplicate_topics()

        # assign documents to topic
        self.doc_top, self.doc_dist = self._calculate_documents_topic(self.topic_vectors, self.document_vectors)

        # calculate topic sizes
        self.topic_sizes = self._calculate_topic_sizes(hierarchy=False)

        # re-order topics
        self._reorder_topics(hierarchy=False)

        # initialize variables for hierarchical topic reduction
        self.topic_vectors_reduced = None
        self.doc_top_reduced = None
        self.doc_dist_reduced = None
        self.topic_sizes_reduced = None
        self.topic_words_reduced = None
        self.topic_word_scores_reduced = None
        self.hierarchy = None

    @staticmethod
    def _l2_normalize(vectors):

        if vectors.ndim == 2:
            return normalize(vectors)
        else:
            return normalize(vectors.reshape(1, -1))[0]

    def _create_topic_vectors(self, cluster_labels):
        """Topic vectors are really just document vectors of """
        unique_labels = set(cluster_labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)
        self.topic_vectors = self._l2_normalize(
            np.vstack([self.document_vectors[np.where(cluster_labels == label)[0]]
                      .mean(axis=0) for label in unique_labels]))

    def _deduplicate_topics(self):
        core_samples, labels = dbscan(X=self.topic_vectors,
                                      eps=0.1,
                                      min_samples=2,
                                      metric="cosine")

        duplicate_clusters = set(labels)

        if len(duplicate_clusters) > 1 or -1 not in duplicate_clusters:

            # unique topics
            unique_topics = self.topic_vectors[np.where(labels == -1)[0]]

            if -1 in duplicate_clusters:
                duplicate_clusters.remove(-1)

            # merge duplicate topics
            for unique_label in duplicate_clusters:
                unique_topics = np.vstack(
                    [unique_topics, self._l2_normalize(self.topic_vectors[np.where(labels == unique_label)[0]]
                                                       .mean(axis=0))])

            self.topic_vectors = unique_topics

    def _calculate_topic_sizes(self, hierarchy=False):
        if hierarchy:
            topic_sizes = pd.Series(self.doc_top_reduced).value_counts()
        else:
            topic_sizes = pd.Series(self.doc_top).value_counts()

        return topic_sizes

    def _reorder_topics(self, hierarchy=False):

        if hierarchy:
            self.topic_vectors_reduced = self.topic_vectors_reduced[self.topic_sizes_reduced.index]
            self.topic_words_reduced = self.topic_words_reduced[self.topic_sizes_reduced.index]
            self.topic_word_scores_reduced = self.topic_word_scores_reduced[self.topic_sizes_reduced.index]
            old2new = dict(zip(self.topic_sizes_reduced.index, range(self.topic_sizes_reduced.index.shape[0])))
            self.doc_top_reduced = np.array([old2new[i] for i in self.doc_top_reduced])
            self.hierarchy = [self.hierarchy[i] for i in self.topic_sizes_reduced.index]
            self.topic_sizes_reduced.reset_index(drop=True, inplace=True)
        else:
            self.topic_vectors = self.topic_vectors[self.topic_sizes.index]
            # self.topic_words = self.topic_words[self.topic_sizes.index]
            # self.topic_word_scores = self.topic_word_scores[self.topic_sizes.index]
            old2new = dict(zip(self.topic_sizes.index, range(self.topic_sizes.index.shape[0])))
            self.doc_top = np.array([old2new[i] for i in self.doc_top])
            self.topic_sizes.reset_index(drop=True, inplace=True)

    @staticmethod
    def _calculate_documents_topic(topic_vectors, document_vectors, dist=True, num_topics=None):
        batch_size = 10000
        doc_top = []
        if dist:
            doc_dist = []

        if document_vectors.shape[0] > batch_size:
            current = 0
            batches = int(document_vectors.shape[0] / batch_size)
            extra = document_vectors.shape[0] % batch_size

            for ind in range(0, batches):
                res = np.inner(document_vectors[current:current + batch_size], topic_vectors)

                if num_topics is None:
                    doc_top.extend(np.argmax(res, axis=1))
                    if dist:
                        doc_dist.extend(np.max(res, axis=1))
                else:
                    doc_top.extend(np.flip(np.argsort(res), axis=1)[:, :num_topics])
                    if dist:
                        doc_dist.extend(np.flip(np.sort(res), axis=1)[:, :num_topics])

                current += batch_size

            if extra > 0:
                res = np.inner(document_vectors[current:current + extra], topic_vectors)

                if num_topics is None:
                    doc_top.extend(np.argmax(res, axis=1))
                    if dist:
                        doc_dist.extend(np.max(res, axis=1))
                else:
                    doc_top.extend(np.flip(np.argsort(res), axis=1)[:, :num_topics])
                    if dist:
                        doc_dist.extend(np.flip(np.sort(res), axis=1)[:, :num_topics])
            if dist:
                doc_dist = np.array(doc_dist)
        else:
            res = np.inner(document_vectors, topic_vectors)

            if num_topics is None:
                doc_top = np.argmax(res, axis=1)
                if dist:
                    doc_dist = np.max(res, axis=1)
            else:
                doc_top.extend(np.flip(np.argsort(res), axis=1)[:, :num_topics])
                if dist:
                    doc_dist.extend(np.flip(np.sort(res), axis=1)[:, :num_topics])

        if num_topics is not None:
            doc_top = np.array(doc_top)
            if dist:
                doc_dist = np.array(doc_dist)

        if dist:
            return doc_top, doc_dist
        else:
            return doc_top

    def search_documents_by_topic(self, topic_num, num_docs, return_documents=True, reduced=False):
        """
        Get the most semantically similar documents to the topic.
        These are the documents closest to the topic vector. Documents are
        ordered by proximity to the topic vector. Successive documents in the
        list are less semantically similar to the topic.
        Parameters
        ----------
        topic_num: int
            The topic number to search.
        num_docs: int
            Number of documents to return.
        return_documents: bool (Optional default True)
            Determines if the documents will be returned. If they were not
            saved in the model they will not be returned.
        reduced: bool (Optional, default False)
            Original topics are used to search by default. If True the
            reduced topics will be used.
        Returns
        -------
        documents: (Optional) array of str, shape(num_docs)
            The documents in a list, the most similar are first.
            Will only be returned if the documents were saved and if
            return_documents is set to True.
        doc_scores: array of float, shape(num_docs)
            Semantic similarity of document to topic. The cosine similarity of
            the document and topic vector.
        doc_ids: array of int, shape(num_docs)
            Unique ids of documents. If ids were not given to the model, the
            index of the document in the model will be returned.
        """

        if reduced:
            self._validate_hierarchical_reduction()
            self._validate_topic_num(topic_num, reduced)
            self._validate_topic_search(topic_num, num_docs, reduced)

            topic_document_indexes = np.where(self.doc_top_reduced == topic_num)[0]
            topic_document_indexes_ordered = np.flip(np.argsort(self.doc_dist_reduced[topic_document_indexes]))
            doc_indexes = topic_document_indexes[topic_document_indexes_ordered][0:num_docs]
            doc_scores = self.doc_dist_reduced[doc_indexes]
            doc_ids = self._get_document_ids(doc_indexes)

        else:

            self._validate_topic_num(topic_num, reduced)
            self._validate_topic_search(topic_num, num_docs, reduced)

            topic_document_indexes = np.where(self.doc_top == topic_num)[0]
            topic_document_indexes_ordered = np.flip(np.argsort(self.doc_dist[topic_document_indexes]))
            doc_indexes = topic_document_indexes[topic_document_indexes_ordered][0:num_docs]
            doc_scores = self.doc_dist[doc_indexes]
            doc_ids = self._get_document_ids(doc_indexes)

        if self.abstracts is not None and return_documents:
            abstracts = self.abstracts[doc_indexes]
            return abstracts, doc_scores, doc_ids
        else:
            return doc_scores, doc_ids

    def _get_document_ids(self, doc_index):
        return self.document_ids[doc_index]

    def _get_document_indexes(self, doc_ids):
        if self.document_ids is None:
            return doc_ids
        else:
            return [self.doc_id2index[doc_id] for doc_id in doc_ids]

    @staticmethod
    def _less_than_zero(num, var_name):
        if num < 0:
            raise ValueError(f"{var_name} cannot be less than 0.")

    def _validate_hierarchical_reduction(self):
        if self.hierarchy is None:
            raise ValueError("Hierarchical topic reduction has not been performed.")

    def _validate_hierarchical_reduction_num_topics(self, num_topics):
        current_num_topics = len(self.topic_vectors)
        if num_topics >= current_num_topics:
            raise ValueError(f"Number of topics must be less than {current_num_topics}.")

    def _validate_num_docs(self, num_docs):
        self._less_than_zero(num_docs, "num_docs")
        document_count = len(self.doc_top)
        if num_docs > document_count:
            raise ValueError(f"num_docs cannot exceed the number of documents: {document_count}.")

    def _validate_num_topics(self, num_topics, reduced):
        self._less_than_zero(num_topics, "num_topics")
        if reduced:
            topic_count = len(self.topic_vectors_reduced)
            if num_topics > topic_count:
                raise ValueError(f"num_topics cannot exceed the number of reduced topics: {topic_count}.")
        else:
            topic_count = len(self.topic_vectors)
            if num_topics > topic_count:
                raise ValueError(f"num_topics cannot exceed the number of topics: {topic_count}.")

    def _validate_topic_num(self, topic_num, reduced):
        self._less_than_zero(topic_num, "topic_num")

        if reduced:
            topic_count = len(self.topic_vectors_reduced) - 1
            if topic_num > topic_count:
                raise ValueError(f"Invalid topic number: valid reduced topics numbers are 0 to {topic_count}.")
        else:
            topic_count = len(self.topic_vectors) - 1
            if topic_num > topic_count:
                raise ValueError(f"Invalid topic number: valid original topics numbers are 0 to {topic_count}.")

    def _validate_topic_search(self, topic_num, num_docs, reduced):
        self._less_than_zero(num_docs, "num_docs")
        if reduced:
            if num_docs > self.topic_sizes_reduced[topic_num]:
                raise ValueError(f"Invalid number of documents: reduced topic {topic_num}"
                                 f" only has {self.topic_sizes_reduced[topic_num]} documents.")
        else:
            if num_docs > self.topic_sizes[topic_num]:
                raise ValueError(f"Invalid number of documents: original topic {topic_num}"
                                 f" only has {self.topic_sizes[topic_num]} documents.")

    def _validate_doc_ids(self, doc_ids, doc_ids_neg):

        if not (isinstance(doc_ids, list) or isinstance(doc_ids, np.ndarray)):
            raise ValueError("doc_ids must be a list of string or int.")
        if not (isinstance(doc_ids_neg, list) or isinstance(doc_ids_neg, np.ndarray)):
            raise ValueError("doc_ids_neg must be a list of string or int.")

        if isinstance(doc_ids, np.ndarray):
            doc_ids = list(doc_ids)
        if isinstance(doc_ids_neg, np.ndarray):
            doc_ids_neg = list(doc_ids_neg)

        doc_ids_all = doc_ids + doc_ids_neg

        if self.document_ids is not None:
            for doc_id in doc_ids_all:
                if doc_id not in self.doc_id2index:
                    raise ValueError(f"{doc_id} is not a valid document id.")
        elif min(doc_ids) < 0:
            raise ValueError(f"{min(doc_ids)} is not a valid document id.")
        elif max(doc_ids) > len(self.doc_top) - 1:
            raise ValueError(f"{max(doc_ids)} is not a valid document id.")

    def _validate_keywords(self, keywords, keywords_neg):
        if not (isinstance(keywords, list) or isinstance(keywords, np.ndarray)):
            raise ValueError("keywords must be a list of strings.")

        if not (isinstance(keywords_neg, list) or isinstance(keywords_neg, np.ndarray)):
            raise ValueError("keywords_neg must be a list of strings.")

        keywords_lower = [keyword.lower() for keyword in keywords]
        keywords_neg_lower = [keyword.lower() for keyword in keywords_neg]

        vocab = self.vocab
        for word in keywords_lower + keywords_neg_lower:
            if word not in vocab:
                raise ValueError(f"'{word}' has not been learned by the model so it cannot be searched.")

        return keywords_lower, keywords_neg_lower

    def _validate_document_ids_add_doc(self, abstracts, document_ids):
        if document_ids is None:
            raise ValueError("Document ids need to be provided.")
        if len(abstracts) != len(document_ids):
            raise ValueError("Document ids need to match number of abstracts.")
        if len(document_ids) != len(set(document_ids)):
            raise ValueError("Document ids need to be unique.")

        if len(set(document_ids).intersection(self.document_ids)) > 0:
            raise ValueError("Some document ids already exist in model.")

        if self.doc_id_type == np.str_:
            if not all((isinstance(doc_id, str) or isinstance(doc_id, np.str_)) for doc_id in document_ids):
                raise ValueError("Document ids need to be of type str.")

        if self.doc_id_type == np.int_:
            if not all((isinstance(doc_id, int) or isinstance(doc_id, np.int_)) for doc_id in document_ids):
                raise ValueError("Document ids need to be of type int.")

    @staticmethod
    def _validate_documents(abstracts):
        if not all((isinstance(doc, str) or isinstance(doc, np.str_)) for doc in abstracts):
            raise ValueError("Abstracts need to be a list of strings.")

    @staticmethod
    def _validate_query(query):
        if not isinstance(query, str) or isinstance(query, np.str_):
            raise ValueError("Query needs to be a string.")

    def _validate_vector(self, vector):
        if not isinstance(vector, np.ndarray):
            raise ValueError("Vector needs to be a numpy array.")
        vec_size = self.document_vectors.shape[1]
        if not vector.shape[0] == vec_size:
            raise ValueError(f"Vector needs to be of {vec_size} dimensions.")

    def get_num_topics(self, reduced=False):
        """
        Get number of topics.
        This is the number of topics Top2Vec has found in the data by default.
        If reduced is True, the number of reduced topics is returned.
        Parameters
        ----------
        reduced: bool (Optional, default False)
            The number of original topics will be returned by default. If True
            will return the number of reduced topics, if hierarchical topic
            reduction has been performed.
        Returns
        -------
        num_topics: int
        """

        if reduced:
            self._validate_hierarchical_reduction()
            return len(self.topic_vectors_reduced)
        else:
            return len(self.topic_vectors)

    def get_topic_sizes(self, reduced=False):
        """
        Get topic sizes.
        The number of documents most similar to each topic. Topics are
        in increasing order of size.
        The sizes of the original topics is returned unless reduced=True,
        in which case the sizes of the reduced topics will be returned.
        Parameters
        ----------
        reduced: bool (Optional, default False)
            Original topic sizes are returned by default. If True the
            reduced topic sizes will be returned.
        Returns
        -------
        topic_sizes: array of int, shape(num_topics)
            The number of documents most similar to the topic.
        topic_nums: array of int, shape(num_topics)
            The unique number of every topic will be returned.
        """
        if reduced:
            self._validate_hierarchical_reduction()
            return np.array(self.topic_sizes_reduced.values), np.array(self.topic_sizes_reduced.index)
        else:
            return np.array(self.topic_sizes.values), np.array(self.topic_sizes.index)

    def get_topic_hierarchy(self):
        """
        Get the hierarchy of reduced topics. The mapping of each original topic
        to the reduced topics is returned.
        Hierarchical topic reduction must be performed before calling this
        method.
        Returns
        -------
        hierarchy: list of ints
            Each index of the hierarchy corresponds to the topic number of a
            reduced topic. For each reduced topic the topic numbers of the
            original topics that were merged to create it are listed.
            Example:
            [[3]  <Reduced Topic 0> contains original Topic 3
            [2,4] <Reduced Topic 1> contains original Topics 2 and 4
            [0,1] <Reduced Topic 3> contains original Topics 0 and 1
            ...]
        """

        self._validate_hierarchical_reduction()

        return self.hierarchy

    def hierarchical_topic_reduction(self, num_topics):
        """
        Reduce the number of topics discovered by Top2Vec.
        The most representative topics of the corpus will be found, by
        iteratively merging each smallest topic to the most similar topic until
        num_topics is reached.
        Parameters
        ----------
        num_topics: int
            The number of topics to reduce to.
        Returns
        -------
        hierarchy: list of ints
            Each index of hierarchy corresponds to the reduced topics, for each
            reduced topic the indexes of the original topics that were merged
            to create it are listed.
            Example:
            [[3]  <Reduced Topic 0> contains original Topic 3
            [2,4] <Reduced Topic 1> contains original Topics 2 and 4
            [0,1] <Reduced Topic 3> contains original Topics 0 and 1
            ...]
        """
        self._validate_hierarchical_reduction_num_topics(num_topics)

        num_topics_current = self.topic_vectors.shape[0]
        top_vecs = self.topic_vectors
        top_sizes = [self.topic_sizes[i] for i in range(0, len(self.topic_sizes))]
        hierarchy = [[i] for i in range(self.topic_vectors.shape[0])]

        count = 0
        interval = max(int(self.document_vectors.shape[0] / 50000), 1)

        while num_topics_current > num_topics:

            # find smallest and most similar topics
            smallest = np.argmin(top_sizes)
            res = np.inner(top_vecs[smallest], top_vecs)
            sims = np.flip(np.argsort(res))
            most_sim = sims[1]
            if most_sim == smallest:
                most_sim = sims[0]

            # calculate combined topic vector
            top_vec_smallest = top_vecs[smallest]
            smallest_size = top_sizes[smallest]

            top_vec_most_sim = top_vecs[most_sim]
            most_sim_size = top_sizes[most_sim]

            combined_vec = self._l2_normalize(((top_vec_smallest * smallest_size) +
                                               (top_vec_most_sim * most_sim_size)) / (smallest_size + most_sim_size))

            # update topic vectors
            ix_keep = list(range(len(top_vecs)))
            ix_keep.remove(smallest)
            ix_keep.remove(most_sim)
            top_vecs = top_vecs[ix_keep]
            top_vecs = np.vstack([top_vecs, combined_vec])
            num_topics_current = top_vecs.shape[0]

            # update topics sizes
            if count % interval == 0:
                doc_top = self._calculate_documents_topic(topic_vectors=top_vecs,
                                                          document_vectors=self.document_vectors,
                                                          dist=False)
                topic_sizes = pd.Series(doc_top).value_counts()
                top_sizes = [topic_sizes[i] for i in range(0, len(topic_sizes))]

            else:
                smallest_size = top_sizes.pop(smallest)
                if most_sim < smallest:
                    most_sim_size = top_sizes.pop(most_sim)
                else:
                    most_sim_size = top_sizes.pop(most_sim - 1)
                combined_size = smallest_size + most_sim_size
                top_sizes.append(combined_size)

            count += 1

            # update topic hierarchy
            smallest_inds = hierarchy.pop(smallest)
            if most_sim < smallest:
                most_sim_inds = hierarchy.pop(most_sim)
            else:
                most_sim_inds = hierarchy.pop(most_sim - 1)

            combined_inds = smallest_inds + most_sim_inds
            hierarchy.append(combined_inds)

        # re-calculate topic vectors from clusters
        doc_top = self._calculate_documents_topic(topic_vectors=top_vecs,
                                                  document_vectors=self.document_vectors,
                                                  dist=False)
        self.topic_vectors_reduced = self._l2_normalize(np.vstack([self.document_vectors
                                                                   [np.where(doc_top == label)[0]]
                                                                  .mean(axis=0) for label in set(doc_top)]))

        self.hierarchy = hierarchy

        # assign documents to topic
        self.doc_top_reduced, self.doc_dist_reduced = self._calculate_documents_topic(self.topic_vectors_reduced,
                                                                                      self.document_vectors)
        # find topic words and scores
        self.topic_words_reduced, self.topic_word_scores_reduced = self._find_topic_words_and_scores(
            topic_vectors=self.topic_vectors_reduced)

        # calculate topic sizes
        self.topic_sizes_reduced = self._calculate_topic_sizes(hierarchy=True)

        # re-order topics
        self._reorder_topics(hierarchy=True)

        return self.hierarchy