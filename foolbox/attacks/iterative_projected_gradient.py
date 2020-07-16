# @Shitong Zhu, SRA 2019
# This Foolbox module has been modified to reflect our "augmented"
# projection algorithm
# Please replace the "iterative_projected_gradient.py" file under
# foolbox Python library directory

from __future__ import division
import numpy as np
from abc import abstractmethod
import logging
import warnings

# for computing distance after each projection
from scipy.spatial import distance

from .base import Attack
from .base import call_decorator
from .. import distances
from ..utils import crossentropy
from .. import nprng

from ..distances import Distance
from ..distances import MSE

from ..map_back import modify_json
from ..map_back import get_file_names
from ..map_back import compute_x_after_mapping_back
from ..map_back import mapback

from ..classify import predict
from ..classify import setup_clf

import matplotlib.pyplot as plt
import math

from urllib.parse import urlsplit

LABLE = {"AD": 1, "NONAD": 0}
NORM_MAP = {
    321: 59,
    322: 60,
    323: 61,
    325: 63,
    326: 64,
}


class IterativeProjectedGradientBaseAttack(Attack):
    """Base class for iterative (projected) gradient attacks.

    Concrete subclasses should implement __call__, _gradient
    and _clip_perturbation.

    TODO: add support for other loss-functions, e.g. the CW loss function,
    see https://github.com/MadryLab/mnist_challenge/blob/master/pgd_attack.py
    """

    @abstractmethod
    def _gradient(self, a, x, class_, strict=True):
        raise NotImplementedError

    @abstractmethod
    def _clip_perturbation(self, a, noise, epsilon):
        raise NotImplementedError

    @abstractmethod
    def _check_distance(self, a):
        raise NotImplementedError

    def _compute_l2_distance(self, x1, x2):
        return distance.cdist(x1, x2, 'euclidean')

    def _compute_lp_distance(self, x1, x2):
        diff = np.array(x1) - np.array(x2)
        value = np.max(np.abs(diff)).astype(np.float64)
        return value

    def _generate_constrained_perturbation(self, perturbation,
                                           perturbable_idx_set,
                                           only_increase_idx_set,
                                           debug=False):

        for i in range(len(perturbation)):
            if i not in perturbable_idx_set:
                perturbation[i] = 0.0
            if i in only_increase_idx_set and perturbation[i] < 0.0:
                perturbation[i] = -perturbation[i]
        return perturbation

    def _unscale_feature(self, val, stats, is_float=False):
        [maxn, minn] = stats
        maxn, minn = float(maxn), float(minn)
        if is_float:
            return val * (maxn + minn) + minn
        else:
            # print(val, maxn, minn)
            return int(round(val * (maxn + minn) + minn))

    def _rescale_feature(self, val, stats):
        [maxn, minn] = stats
        maxn, minn = float(maxn), float(minn)
        if maxn == minn:
            return val
        return (val - minn) / (maxn - minn)

    def _calculate_diff(self, ori, per, stats):
        return self._unscale_feature(per, stats) - self._unscale_feature(ori, stats)

    def _recalculate_related_features(self, candidate,
                                      original,
                                      normalization_ratios,
                                      perturbable_idx_set,
                                      debug=False):

        recalculated_candidate = np.copy(candidate)

        should_process_global_ratio = False
        should_process_connection_cnt = False
        should_match_edge_to_node = False
        should_process_sibling_cnt = False

        if perturbable_idx_set is not None and normalization_ratios is not None:
            if 0 in perturbable_idx_set or 1 in perturbable_idx_set:
                should_process_global_ratio = True
            if 0 in perturbable_idx_set and 1 not in perturbable_idx_set:
                should_match_edge_to_node = True
            if 4 in perturbable_idx_set or 5 in perturbable_idx_set:
                should_process_connection_cnt = True
            if 22 in perturbable_idx_set:
                should_process_sibling_cnt = True

        if should_process_sibling_cnt:
            sibling_cnt_diff = self._calculate_diff(
                original[22], recalculated_candidate[22], normalization_ratios[22]['val'])
            node_cnt_diff = self._calculate_diff(original[0], recalculated_candidate[0], normalization_ratios[0]['val'])
            if node_cnt_diff <= sibling_cnt_diff:
                recalculated_candidate[0] += self._rescale_feature(sibling_cnt_diff, normalization_ratios[22]['val'])

        if should_process_connection_cnt:
            original_5th_feature = self._unscale_feature(
                recalculated_candidate[4], normalization_ratios[4]['val'], False)
            original_6th_feature = self._unscale_feature(
                recalculated_candidate[5], normalization_ratios[5]['val'], False)
            recalculated_candidate[6] = self._rescale_feature(
                original_5th_feature + original_6th_feature, normalization_ratios[6]['val'])

        if should_match_edge_to_node:
            # print(original[0], recalculated_candidate[0], normalization_ratios[0]['val'])
            node_cnt_diff = self._calculate_diff(original[0], recalculated_candidate[0], normalization_ratios[0]['val'])
            recalculated_candidate[1] += self._rescale_feature(node_cnt_diff, normalization_ratios[1]['val'])

        if should_process_global_ratio:
            original_1st_feature = self._unscale_feature(
                recalculated_candidate[0], normalization_ratios[0]['val'], False)
            original_2nd_feature = self._unscale_feature(
                recalculated_candidate[1], normalization_ratios[1]['val'], False)
            if original_1st_feature != 0.0 and original_2nd_feature != 0.0:
                recalculated_candidate[2] = self._rescale_feature(
                    original_1st_feature / original_2nd_feature, normalization_ratios[2]['val'])
                recalculated_candidate[3] = self._rescale_feature(
                    original_2nd_feature / original_1st_feature, normalization_ratios[3]['val'])
            if debug:
                print("original_1st_feature:", original_1st_feature)
                print("original_2nd_feature:", original_2nd_feature)
                print("recalculated_candidate[2]", recalculated_candidate[2])
                print("recalculated_candidate[3]", recalculated_candidate[3])
                print("normalization_ratios[3]['val']", normalization_ratios[3]['val'])
                print("normalization_ratios[2]['val']", normalization_ratios[2]['val'])
                print("original_2nd_feature / original_1st_feature", original_2nd_feature / original_1st_feature)
                print("float(original_2nd_feature / original_1st_feature)",
                      float(original_2nd_feature / original_1st_feature))

        return recalculated_candidate

    def _reject_imperturbable_features(self, candidate,
                                       original, perturbable_idx_set,
                                       debug=False):

        assert len(candidate) == len(original), "[ERROR] Lengths of two input arrays not equal!"

        rejected_candidate = []

        for i in range(len(candidate)):
            if i in perturbable_idx_set:
                rejected_candidate.append(candidate[i])
            else:
                rejected_candidate.append(original[i])

        return np.array(rejected_candidate)

    def _reject_only_increase_features(self, candidate,
                                       original, only_increase_idx_set,
                                       debug=False):

        assert len(candidate) == len(original), "[ERROR] Lengths of two input arrays not equal!"

        rejected_candidate = []

        for i in range(len(candidate)):
            if i in only_increase_idx_set and candidate[i] < original[i]:
                rejected_candidate.append(original[i])
            else:
                rejected_candidate.append(candidate[i])

        return np.array(rejected_candidate)

    def _legalize_candidate(self, candidate,
                            original,
                            feature_types,
                            debug=False):
        if debug:
            print("[INFO] Entered candidate legalization method!")
            print(feature_types)
            input("Press Enter to continue...")
        adv_x = np.copy(candidate)
        processed_adv_x = np.copy(candidate)

        for i in range(len(adv_x)):
            adv_val = adv_x[i]
            ori_val = original[i]
            processed_adv_val = None
            if feature_types[i] == 'b':
                if abs(adv_val - 1.0) > abs(adv_val - 0.0):
                    processed_adv_val = 1
                else:
                    processed_adv_val = 0

            if processed_adv_val is not None:
                processed_adv_x[i] = processed_adv_val

        lookahead_cnt = 0
        for i in range(len(adv_x)):
            if lookahead_cnt - 1 > 0:
                lookahead_cnt -= 1
                continue
            if feature_types[i] == 'c':
                j = i
                while feature_types[j] == 'c' and j + 1 < len(adv_x):
                    j += 1
                categorical_interval_end = j
                maxn = -10000
                maxn_idx = i
                for j in range(i, categorical_interval_end):
                    if adv_x[j] > maxn:
                        maxn = adv_x[j]
                        maxn_idx = j
                for j in range(i, categorical_interval_end):
                    if j == maxn_idx:
                        processed_adv_val = 1
                    else:
                        processed_adv_val = 0
                    processed_adv_x[j] = processed_adv_val
                    lookahead_cnt += 1

        legalized_candidate = processed_adv_x

        return legalized_candidate

    def _get_mode_and_class(self, a):
        # determine if the attack is targeted or not
        target_class = a.target_class()
        targeted = target_class is not None

        if targeted:
            class_ = target_class
        else:
            class_ = a.original_class
        return targeted, class_

    def _run(self, a, binary_search,
             epsilon, stepsize, iterations,
             random_start, return_early,
             perturbable_idx_set, only_increase_idx_set,
             feature_defs, normalization_ratios,
             enforce_interval, request_id, model,
             map_back_mode, logger):
        if not a.has_gradient():
            warnings.warn('applied gradient-based attack to model that'
                          ' does not provide gradients')
            return

        self._check_distance(a)

        targeted, class_ = self._get_mode_and_class(a)

        if binary_search:
            if isinstance(binary_search, bool):
                k = 20
            else:
                k = int(binary_search)
            return self._run_binary_search(
                a, epsilon, stepsize, iterations,
                random_start, targeted, class_, return_early, k=k,
                perturbable_idx_set=perturbable_idx_set,
                only_increase_idx_set=only_increase_idx_set,
                feature_defs=feature_defs,
                normalization_ratios=normalization_ratios,
                enforce_interval=enforce_interval,
                request_id=request_id)
        else:
            return self._run_one(
                a, epsilon, stepsize, iterations,
                random_start, targeted, class_, return_early,
                perturbable_idx_set=perturbable_idx_set,
                only_increase_idx_set=only_increase_idx_set,
                feature_defs=feature_defs,
                normalization_ratios=normalization_ratios,
                enforce_interval=enforce_interval,
                request_id=request_id, model=model, map_back_mode=map_back_mode,
                logger=logger)

    def _get_geometric_enforce_interval(self, base, iteations, i):
        progress = float(i + 1) / iteations
        if progress < 0.1:
            growth_ratio = 1
        elif progress >= 0.1 and progress < 0.5:
            growth_ratio = 3
        else:
            growth_ratio = 10
        grown = base * growth_ratio
        return grown

    def _run_binary_search(self, a, epsilon, stepsize, iterations,
                           random_start, targeted, class_, return_early, k,
                           perturbable_idx_set, only_increase_idx_set, feature_defs,
                           normalization_ratios, enforce_interval, request_id):

        factor = stepsize / epsilon

        def try_epsilon(epsilon):
            stepsize = factor * epsilon
            return self._run_one(
                a, epsilon, stepsize, iterations,
                random_start, targeted, class_, return_early,
                perturbable_idx_set, only_increase_idx_set, feature_defs,
                normalization_ratios, enforce_interval, request_id)

        for i in range(k):
            if try_epsilon(epsilon):
                logging.info('successful for eps = {}'.format(epsilon))
                break
            logging.info('not successful for eps = {}'.format(epsilon))
            epsilon = epsilon * 1.5
        else:
            logging.warning('exponential search failed')
            return

        bad = 0
        good = epsilon

        for i in range(k):
            epsilon = (good + bad) / 2
            if try_epsilon(epsilon):
                good = epsilon
                logging.info('successful for eps = {}'.format(epsilon))
            else:
                bad = epsilon
                logging.info('not successful for eps = {}'.format(epsilon))

    def _get_diff(self, x, original, perturbable_idx_set, normalization_ratios,
                  debug=False):
        if debug:
            print(original[0], x[0])
            print(original[1], x[1])

        delta = {}
        for idx in list(perturbable_idx_set):
            if normalization_ratios[idx]['type'] == "B":
                diff = float(x[idx]) - float(original[idx])
            else:
                diff = self._calculate_diff(original[idx], x[idx], normalization_ratios[idx]['val'])
            delta[idx] = diff

        return delta

    def _get_x_after_mapping_back(self, domain, url_id, diff, mapback_mode, working_dir="~/Desktop/AdGraphAPI/scripts", first_time=False):
        original_json, original_json_fname, url_id_map = get_file_names(domain)
        if mapback_mode == "centralized":
            modified_json = modify_json(original_json, domain, url_id, diff, url_id_map, original_json_fname)
        if mapback_mode == "distributed":
            request_url = url_id_map[url_id]
            modified_json = mapback(original_json, diff, request_url)
        mapped_x, mapped_unnormalized_x = compute_x_after_mapping_back(
            domain, url_id, modified_json, original_json_fname, working_dir=working_dir, first_time=first_time)
        return mapped_x, mapped_unnormalized_x

    def _deprocess_x(self, x, feature_types, verbal=False):
        FEATURE_TYPES = {'F', 'B', 'C', 'S', 'D', 'L', 'FF'}

        def deprocess_float_feature(val, ratio, ffloat=True):
            val = float(val)
            maxn = float(ratio[0])
            minn = float(ratio[1])
            if ffloat:
                return float(val * (maxn - minn) + minn)
            else:
                return round(val * (maxn - minn) + minn)

        def deprocess_nominal_feature(val, category_name):
            val = float(val)
            if val == 1.0:
                return category_name
            elif val == 0.0:
                return None
            else:
                print("[ERROR] WTF? val: %s %s" % (str(val), category_name))
                raise Exception

        def deprocess_shift_feature(val, offset):
            val = float(val)
            offset = float(offset)
            return int(math.ceil(val + offset))

        features = x
        deprocessed_features = []

        for j in range(len(feature_types)):
            assert feature_types[j]['type'] in FEATURE_TYPES, "[ERROR] Feature type not supported!"

            if feature_types[j]['type'] == 'F':
                ratio = feature_types[j]['val']
                deprocessed_features.append(
                    deprocess_float_feature(features[j], ratio, ffloat=False))

            if feature_types[j]['type'] == 'FF':
                ratio = feature_types[j]['val']
                deprocessed_features.append(
                    deprocess_float_feature(features[j], ratio, ffloat=True))

            if feature_types[j]['type'] == 'C':
                category_name = feature_types[j]['val']
                new_val = deprocess_nominal_feature(features[j], category_name)
                if new_val:
                    deprocessed_features.append(new_val)

            if feature_types[j]['type'] == 'S':
                offset = feature_types[j]['val']
                deprocessed_features.append(
                    deprocess_shift_feature(features[j], offset))

            if feature_types[j]['type'] == 'B':
                val = features[j]
                deprocessed_features.append(int(float(val)))

            if feature_types[j]['type'] == 'D':
                val = features[j]
                deprocessed_features.append(val)

            # label column
            if feature_types[j]['type'] == 'L':
                label = features[j]
                deprocessed_features.append(label)

        return deprocessed_features

    def _compare_x(self, x1, x2, tags=["X1", "X2"], it=None, X3=None):
        assert len(x1) == len(x2), "[ERROR] Two inputs have different sizes!"
        if it is not None:
            print("Iter #%d" % it)
        for i in range(len(x1)):
            if x1[i] != x2[i]:
                if X3 is not None:
                    print("i:", i, tags[0], ":", x1[i], tags[1], ":", x2[i], "original:", X3[i])
                else:
                    print("i:", i, tags[0], ":", x1[i], tags[1], ":", x2[i])

    def _retrain_local_model(self, model, x, y):
        def __generate_mini_batch(x, y, whole_train_set):
            # import random
            train_set = whole_train_set.sample(n=50)
            train_y = train_set.pop('CLASS').to_numpy().tolist()
            train_x = train_set.to_numpy().tolist()
            for _ in range(50):
                train_x.append(x)
                train_y.append(y)
            return np.array(train_x), np.array(train_y)

        train_x, train_y = __generate_mini_batch(x, y, self._train_data_set)
        model.fit(
            x=train_x,
            y=train_y,
            batch_size=100,
            epochs=1
        )

    def _get_label(self, logits):
        if logits[1] > logits[0]:
            return "AD"
        else:
            return "NONAD"

    def _run_one(self, a, epsilon, stepsize, iterations,
                 random_start, targeted, class_, return_early, model,
                 perturbable_idx_set=None, only_increase_idx_set=None,
                 feature_defs=None, normalization_ratios=None,
                 enforce_interval=1, request_id="URL_dummy", dynamic_interval=False,
                 debug=False, draw=False, map_back_mode=True, remote_model=True,
                 check_correctness=False, check_init_correctness=True, logger=None):
        if draw:
            x_axis, y_l2, y_lf = [], [], []

        domain, url_id = request_id.split(',')
        domain = "{0.netloc}".format(urlsplit(domain))

        min_, max_ = a.bounds()
        s = max_ - min_

        original = a.unperturbed.copy()

        if random_start:
            # using uniform noise even if the perturbation clipping uses
            # a different norm because cleverhans does it the same way
            noise = nprng.uniform(
                -epsilon * s, epsilon * s, original.shape).astype(
                    original.dtype)
            x = original + self._clip_perturbation(a, noise, epsilon)
            strict = False  # because we don't enforce the bounds here
        else:
            x = original
            strict = True  # we don't care about the bounds because we are not attacking image clf

        success = False
        success_cent, success_dist = False, False

        if enforce_interval == iterations - 1:
            only_post_process = True
        else:
            only_post_process = False

        is_first_iter = True
        for i in range(iterations):

            if dynamic_interval:
                curr_interval = self._get_geometric_enforce_interval(enforce_interval, iterations, i)
            else:
                curr_interval = enforce_interval

            if i != 0 and (i % curr_interval == 0 or i == iterations - 1):
                should_enforce_policy = True
            else:
                should_enforce_policy = False

            gradient = self._gradient(a, x, class_, strict=strict)
            # non-strict only for the first call and
            # only if random_start is True
            if targeted:
                gradient = -gradient

            # untargeted: gradient ascent on cross-entropy to original class
            # targeted: gradient descent on cross-entropy to target class
            # (this is the actual attack step)
            x = x + stepsize * gradient

            if should_enforce_policy:
                x_before_projection = x.copy()

            # phase 1: reject disallowed perturbations by changing feature values
            # back to original
            if only_post_process:
                if should_enforce_policy and perturbable_idx_set is not None:
                    x = self._reject_imperturbable_features(
                        x,
                        original,
                        perturbable_idx_set)
            else:
                if perturbable_idx_set is not None:
                    x = self._reject_imperturbable_features(
                        x,
                        original,
                        perturbable_idx_set)

            # phase 1: reject decreasing perturbations by changing only-increase
            # feature values back to original
            if should_enforce_policy and only_increase_idx_set is not None:
                x = self._reject_only_increase_features(
                    x,
                    original,
                    only_increase_idx_set)

            if only_post_process:
                if should_enforce_policy and normalization_ratios is not None:
                    x = self._recalculate_related_features(
                        x,
                        original,
                        normalization_ratios,
                        perturbable_idx_set)
            else:
                if normalization_ratios is not None:
                    x = self._recalculate_related_features(
                        x,
                        original,
                        normalization_ratios,
                        perturbable_idx_set)

            # phase 2: change values back to allowed ranges
            if should_enforce_policy and feature_defs is not None:
                x = self._legalize_candidate(
                    x,
                    original,
                    feature_defs)

            if should_enforce_policy:
                l2_dist = self._compute_l2_distance([x_before_projection], [x])
                lf_dist = self._compute_lp_distance([x_before_projection], [x])
                if draw:
                    x_axis.append(i)
                    y_l2.append(l2_dist[0])
                    y_lf.append(lf_dist)
                if debug:
                    print("Step #%d, L2 distance: %f | LP distance: %f" % (i, l2_dist, lf_dist))

            x = original + self._clip_perturbation(a, x - original, original, epsilon, feature_defs)
            x = np.clip(x, min_, max_)

            if should_enforce_policy and map_back_mode:
                diff = self._get_diff(
                    x,
                    original,
                    perturbable_idx_set,
                    normalization_ratios)
                print("Delta at iter #%d: %s" % (i, str(diff)))
                x_before_mapping_back = x.copy()
                x_cent, unnorm_x_cent = self._get_x_after_mapping_back(
                    domain, url_id, diff, "centralized", first_time=is_first_iter)
                x_dist, unnorm_x_dist = self._get_x_after_mapping_back(
                    domain, url_id, diff, "distributed", first_time=is_first_iter)
                is_first_iter = False

                del unnorm_x_cent[-1]  # remove label
                del unnorm_x_dist[-1]  # remove label

                for j in range(len(x_cent)):
                    if j in diff:
                        if j not in NORM_MAP:
                            continue
                        if diff[j] == 1.0:
                            x_cent[j] = 1.0
                            x_dist[j] = 1.0
                            unnorm_x_cent[NORM_MAP[j]] = "1"
                            unnorm_x_dist[NORM_MAP[j]] = "1"
                        if diff[j] == -1.0:
                            x_cent[j] = 0.0
                            x_dist[j] = 0.0
                            unnorm_x_cent[NORM_MAP[j]] = "0"
                            unnorm_x_dist[NORM_MAP[j]] = "0"

                print("unnorm_x_cent:", unnorm_x_cent)
                print("unnorm_x_dist:", unnorm_x_dist)

            # if should_enforce_policy:
            #     if map_back_mode:
            #         x_cent = original + self._clip_perturbation(a, x_cent - original, epsilon, feature_defs)
            #         x_dist = original + self._clip_perturbation(a, x_dist - original, epsilon, feature_defs)
            #     else:
            #         x = original + self._clip_perturbation(a, x - original, epsilon, feature_defs)

            # Call order changed to ensure the perturbed x is
            # still within the bounds
            # if should_enforce_policy:
            #     if map_back_mode:
            #         x_cent = np.clip(x_cent, min_, max_)
            #         x_dist = np.clip(x_dist, min_, max_)
            #     else:
            #         x = np.clip(x, min_, max_)

            if should_enforce_policy and remote_model and not map_back_mode:
                unnorm_x = self._deprocess_x(x, normalization_ratios)
                unnorm_unperturbed = self._deprocess_x(original, normalization_ratios)

            if i == 0 and check_init_correctness:
                unnorm_unperturbed = self._deprocess_x(original, normalization_ratios)
                prediction_unperturbed_local = self._get_label(model.predict(np.array([original]))[0])
                prediction_unperturbed_remote = predict(unnorm_unperturbed, self._remote_model)[0]
                if prediction_unperturbed_local == prediction_unperturbed_remote == "AD":
                    print("Sanity check passed!")
                else:
                    print("Local:", prediction_unperturbed_local)
                    print("Remote:", prediction_unperturbed_remote)
                    print("Sanity check failed!")
                    return False

            if should_enforce_policy:
                # logits, is_adversarial = a.forward_one(x)
                if logging.getLogger().isEnabledFor(logging.DEBUG):
                    if targeted:
                        ce = crossentropy(a.original_class, logits)
                        logging.debug('crossentropy to {} is {}'.format(
                            a.original_class, ce))
                    ce = crossentropy(class_, logits)
                    logging.debug('crossentropy to {} is {}'.format(class_, ce))

                # Use X before mapping back as the starting point for next iteration
                # as there might be feature perturbations that we cannot incoperate in
                # feature space. That is, we should only "descend" by changing "perturbable"
                # features in a "legitimate" fashion
                if remote_model:
                    if map_back_mode:
                        prediction_remote_cent = predict(unnorm_x_cent, self._remote_model)[0]
                        prediction_remote_dist = predict(unnorm_x_dist, self._remote_model)[0]
                    else:
                        prediction_remote = predict(unnorm_x, self._remote_model)[0]
                    prediction_original = predict(unnorm_unperturbed, self._remote_model)[0]
                    if map_back_mode:
                        prediction_local_cent = self._get_label(model.predict(np.array([x_cent]))[0])
                        prediction_local_dist = self._get_label(model.predict(np.array([x_dist]))[0])
                    else:
                        prediction_local = self._get_label(model.predict(np.array([x]))[0])

                    if not only_post_process:
                        if map_back_mode:
                            retrain_cnt_cent, retrain_cnt_dist = 0, 0
                            print("Remote (cent): %s / local (cent): %s" %
                                  (prediction_remote_cent, prediction_local_cent))
                            print("Remote (dist): %s / local (dist): %s" %
                                  (prediction_remote_dist, prediction_local_dist))
                            while prediction_remote_cent != prediction_local_cent and retrain_cnt_cent < 10:
                                retrain_cnt_cent += 1
                                self._retrain_local_model(model, x_cent, LABLE[prediction_remote_cent])
                                prediction_local = self._get_label(model.predict(np.array([x_cent]))[0])
                                print("(cent) iter #%d, has retrained %d time(s)" % (i, retrain_cnt_cent))
                            while prediction_remote_dist != prediction_local_dist and retrain_cnt_dist < 10:
                                retrain_cnt_dist += 1
                                self._retrain_local_model(model, x_dist, LABLE[prediction_remote_dist])
                                prediction_local = self._get_label(model.predict(np.array([x_dist]))[0])
                                print("(dist) iter #%d, has retrained %d time(s)" % (i, retrain_cnt_dist))
                        else:
                            retrain_cnt = 0
                            print("Remote: %s / local: %s" % (prediction_remote, prediction_local))
                            while prediction_remote != prediction_local and retrain_cnt < 10:
                                retrain_cnt += 1
                                self._retrain_local_model(model, x, LABLE[prediction_remote])
                                prediction_local = self._get_label(model.predict(np.array([x]))[0])
                                print("iter #%d, has retrained %d time(s)" % (i, retrain_cnt))

                    if map_back_mode:
                        if prediction_original == "AD" and prediction_remote_cent == "NONAD":
                            success_cent = True
                        if prediction_original == "AD" and prediction_remote_dist == "NONAD":
                            success_dist = True

                        if success_cent and not success_dist:
                            msg = "SUCCESS, centralized, iter_%d, %s, %s, %s" % (i, domain, url_id, str(diff))
                            print(msg)
                            logger.info(msg)
                            return True
                        if not success_cent and success_dist:
                            msg = "SUCCESS, distributed, iter_%d, %s, %s, %s" % (i, domain, url_id, str(diff))
                            print(msg)
                            logger.info(msg)
                            return True
                        if success_cent and success_dist:
                            msg = "SUCCESS, both, iter_%d, %s, %s, %s" % (i, domain, url_id, str(diff))
                            print(msg)
                            logger.info(msg)
                            return True

                        x = x_before_mapping_back
                    else:
                        if prediction_original == "AD" and prediction_remote == "NONAD":
                            msg = "SUCCESS, iter_%d, %s, %s, %s" % (i, domain, url_id, str(diff))
                            print(msg)
                            logger.info(msg)
                            return True
        if draw:
            plt.plot(x_axis, y_l2, linewidth=3)
            plt.plot(x_axis, y_lf, linewidth=3)
            plt.show()
        msg = "FAIL: iter_%d, %s, %s, %s" % (i, domain, url_id, str(diff))
        print(msg)
        logger.info(msg)
        return False


class LinfinityGradientMixin(object):
    def _gradient(self, a, x, class_, strict=True):
        gradient = a.gradient_one(x, class_, strict=strict)
        gradient = np.sign(gradient)
        min_, max_ = a.bounds()
        gradient = (max_ - min_) * gradient
        return gradient


class L1GradientMixin(object):
    def _gradient(self, a, x, class_, strict=True):
        gradient = a.gradient_one(x, class_, strict=strict)
        # using mean to make range of epsilons comparable to Linf
        gradient = gradient / np.mean(np.abs(gradient))
        min_, max_ = a.bounds()
        gradient = (max_ - min_) * gradient
        return gradient


class L2GradientMixin(object):
    def _gradient(self, a, x, class_, strict=True):
        gradient = a.gradient_one(x, class_, strict=strict)
        # using mean to make range of epsilons comparable to Linf
        gradient = gradient / np.sqrt(np.mean(np.square(gradient)))
        min_, max_ = a.bounds()
        gradient = (max_ - min_) * gradient
        return gradient


class LinfinityClippingMixin(object):
    def _clip_perturbation(self, a, perturbation, original, epsilon, feature_defs):
        _LOCAL_EPSILON = 0.8  # hard-coded for now
        _LOCAL_FEATURE_IDX_SET = {0, 1}  # hard-coded for now as well

        original_perturbation = perturbation.copy()
        min_, max_ = a.bounds()
        s = max_ - min_
        clipped = np.clip(perturbation, -epsilon * s, epsilon * s)

        for i in range(len(clipped)):
            if feature_defs[i] in {"b", "c"}:
                clipped[i] = original_perturbation[i]
            if i in _LOCAL_FEATURE_IDX_SET:
                s_local = original[i]
                offset = min(epsilon * s, _LOCAL_EPSILON * s_local)  # use the smaller offset
                clipped[i] = np.clip([original_perturbation[i]], -offset, offset)[0]

        return clipped


class L1ClippingMixin(object):
    def _clip_perturbation(self, a, perturbation, epsilon):
        # using mean to make range of epsilons comparable to Linf
        norm = np.mean(np.abs(perturbation))
        norm = max(1e-12, norm)  # avoid divsion by zero
        min_, max_ = a.bounds()
        s = max_ - min_
        # clipping, i.e. only decreasing norm
        factor = min(1, epsilon * s / norm)
        return perturbation * factor


class L2ClippingMixin(object):
    def _clip_perturbation(self, a, perturbation, epsilon):
        # using mean to make range of epsilons comparable to Linf
        norm = np.sqrt(np.mean(np.square(perturbation)))
        norm = max(1e-12, norm)  # avoid divsion by zero
        min_, max_ = a.bounds()
        s = max_ - min_
        # clipping, i.e. only decreasing norm
        factor = min(1, epsilon * s / norm)
        # input(str(perturbation * factor))
        return perturbation * factor


class LinfinityDistanceCheckMixin(object):
    def _check_distance(self, a):
        if not isinstance(a.distance, distances.Linfinity):
            logging.warning('Running an attack that tries to minimize the'
                            ' Linfinity norm of the perturbation without'
                            ' specifying foolbox.distances.Linfinity as'
                            ' the distance metric might lead to suboptimal'
                            ' results.')


class L1DistanceCheckMixin(object):
    def _check_distance(self, a):
        if not isinstance(a.distance, distances.MAE):
            logging.warning('Running an attack that tries to minimize the'
                            ' L1 norm of the perturbation without'
                            ' specifying foolbox.distances.MAE as'
                            ' the distance metric might lead to suboptimal'
                            ' results.')


class L2DistanceCheckMixin(object):
    def _check_distance(self, a):
        if not isinstance(a.distance, distances.MSE):
            logging.warning('Running an attack that tries to minimize the'
                            ' L2 norm of the perturbation without'
                            ' specifying foolbox.distances.MSE as'
                            ' the distance metric might lead to suboptimal'
                            ' results.')


class LinfinityBasicIterativeAttack(
        LinfinityGradientMixin,
        LinfinityClippingMixin,
        LinfinityDistanceCheckMixin,
        IterativeProjectedGradientBaseAttack):

    """The Basic Iterative Method introduced in [1]_.

    This attack is also known as Projected Gradient
    Descent (PGD) (without random start) or FGMS^k.

    References
    ----------
    .. [1] Alexey Kurakin, Ian Goodfellow, Samy Bengio,
           "Adversarial examples in the physical world",
            https://arxiv.org/abs/1607.02533

    .. seealso:: :class:`ProjectedGradientDescentAttack`

    """

    @call_decorator
    def __call__(self, input_or_adv, label=None, unpack=True,
                 binary_search=True,
                 epsilon=0.3,
                 stepsize=0.05,
                 iterations=10,
                 random_start=False,
                 return_early=True):
        """Simple iterative gradient-based attack known as
        Basic Iterative Method, Projected Gradient Descent or FGSM^k.

        Parameters
        ----------
        input_or_adv : `numpy.ndarray` or :class:`Adversarial`
            The original, unperturbed input as a `numpy.ndarray` or
            an :class:`Adversarial` instance.
        label : int
            The reference label of the original input. Must be passed
            if `a` is a `numpy.ndarray`, must not be passed if `a` is
            an :class:`Adversarial` instance.
        unpack : bool
            If true, returns the adversarial input, otherwise returns
            the Adversarial object.
        binary_search : bool or int
            Whether to perform a binary search over epsilon and stepsize,
            keeping their ratio constant and using their values to start
            the search. If False, hyperparameters are not optimized.
            Can also be an integer, specifying the number of binary
            search steps (default 20).
        epsilon : float
            Limit on the perturbation size; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        stepsize : float
            Step size for gradient descent; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        iterations : int
            Number of iterations for each gradient descent run.
        random_start : bool
            Start the attack from a random point rather than from the
            original input.
        return_early : bool
            Whether an individual gradient descent run should stop as
            soon as an adversarial is found.
        """

        a = input_or_adv
        del input_or_adv
        del label
        del unpack

        assert epsilon > 0

        self._run(a, binary_search,
                  epsilon, stepsize, iterations,
                  random_start, return_early)


BasicIterativeMethod = LinfinityBasicIterativeAttack
BIM = BasicIterativeMethod


class L1BasicIterativeAttack(
        L1GradientMixin,
        L1ClippingMixin,
        L1DistanceCheckMixin,
        IterativeProjectedGradientBaseAttack):

    """Modified version of the Basic Iterative Method
    that minimizes the L1 distance.

    .. seealso:: :class:`LinfinityBasicIterativeAttack`

    """

    @call_decorator
    def __call__(self, input_or_adv, label=None, unpack=True,
                 binary_search=True,
                 epsilon=0.3,
                 stepsize=0.05,
                 iterations=10,
                 random_start=False,
                 return_early=True):
        """Simple iterative gradient-based attack known as
        Basic Iterative Method, Projected Gradient Descent or FGSM^k.

        Parameters
        ----------
        input_or_adv : `numpy.ndarray` or :class:`Adversarial`
            The original, unperturbed input as a `numpy.ndarray` or
            an :class:`Adversarial` instance.
        label : int
            The reference label of the original input. Must be passed
            if `a` is a `numpy.ndarray`, must not be passed if `a` is
            an :class:`Adversarial` instance.
        unpack : bool
            If true, returns the adversarial input, otherwise returns
            the Adversarial object.
        binary_search : bool or int
            Whether to perform a binary search over epsilon and stepsize,
            keeping their ratio constant and using their values to start
            the search. If False, hyperparameters are not optimized.
            Can also be an integer, specifying the number of binary
            search steps (default 20).
        epsilon : float
            Limit on the perturbation size; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        stepsize : float
            Step size for gradient descent; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        iterations : int
            Number of iterations for each gradient descent run.
        random_start : bool
            Start the attack from a random point rather than from the
            original input.
        return_early : bool
            Whether an individual gradient descent run should stop as
            soon as an adversarial is found.
        """

        a = input_or_adv
        del input_or_adv
        del label
        del unpack

        assert epsilon > 0

        self._run(a, binary_search,
                  epsilon, stepsize, iterations,
                  random_start, return_early)


class L2BasicIterativeAttack(
        L2GradientMixin,
        L2ClippingMixin,
        L2DistanceCheckMixin,
        IterativeProjectedGradientBaseAttack):

    """Modified version of the Basic Iterative Method
    that minimizes the L2 distance.

    .. seealso:: :class:`LinfinityBasicIterativeAttack`

    """

    @call_decorator
    def __call__(self, input_or_adv, label=None, unpack=True,
                 binary_search=True,
                 epsilon=0.3,
                 stepsize=0.05,
                 iterations=10,
                 random_start=False,
                 return_early=True):
        """Simple iterative gradient-based attack known as
        Basic Iterative Method, Projected Gradient Descent or FGSM^k.

        Parameters
        ----------
        input_or_adv : `numpy.ndarray` or :class:`Adversarial`
            The original, unperturbed input as a `numpy.ndarray` or
            an :class:`Adversarial` instance.
        label : int
            The reference label of the original input. Must be passed
            if `a` is a `numpy.ndarray`, must not be passed if `a` is
            an :class:`Adversarial` instance.
        unpack : bool
            If true, returns the adversarial input, otherwise returns
            the Adversarial object.
        binary_search : bool or int
            Whether to perform a binary search over epsilon and stepsize,
            keeping their ratio constant and using their values to start
            the search. If False, hyperparameters are not optimized.
            Can also be an integer, specifying the number of binary
            search steps (default 20).
        epsilon : float
            Limit on the perturbation size; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        stepsize : float
            Step size for gradient descent; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        iterations : int
            Number of iterations for each gradient descent run.
        random_start : bool
            Start the attack from a random point rather than from the
            original input.
        return_early : bool
            Whether an individual gradient descent run should stop as
            soon as an adversarial is found.
        """

        # a = input_or_adv
        # del input_or_adv
        # del label
        # del unpack

        # assert epsilon > 0

        # self._run(a, binary_search,
        #           epsilon, stepsize, iterations,
        #           random_start, return_early)

    def _read_dataset(self, fname):
        import pandas as pd
        from sklearn.utils import shuffle

        train_dataframe = pd.read_csv(fname)
        dataframe = shuffle(train_dataframe)

        return dataframe

    # overriden init method in ABC
    def _initialize(self):
        self._remote_model = setup_clf("/home/shitong/Desktop/attack-adgraph-pipeline/model/rf.pkl")
        self._train_data_set = self._read_dataset(
            "/home/shitong/Desktop/attack-adgraph-pipeline/data/hand_preprocessed_trimmed_label_train_gt_augmented_train_set.csv")

    @call_decorator
    def __call__(self, input_or_adv, label=None, unpack=True,
                 binary_search=True,
                 epsilon=0.3,
                 stepsize=0.01,
                 iterations=40,
                 random_start=False,
                 return_early=True,
                 perturbable_idx_set=None,
                 only_increase_idx_set=None,
                 feature_defs=None,
                 normalization_ratios=None,
                 enforce_interval=1,
                 request_id="URL_dummy",
                 model=None,
                 map_back_mode=False):

        a = input_or_adv
        del input_or_adv
        del label
        del unpack

        assert epsilon > 0

        print("Parameters:", epsilon, stepsize, iterations, enforce_interval, request_id)
        self._run(a, binary_search,
                  epsilon, stepsize, iterations,
                  random_start, return_early,
                  perturbable_idx_set, only_increase_idx_set,
                  feature_defs, normalization_ratios,
                  enforce_interval, request_id,
                  model, map_back_mode)


class ProjectedGradientDescentAttack(
        LinfinityGradientMixin,
        LinfinityClippingMixin,
        LinfinityDistanceCheckMixin,
        IterativeProjectedGradientBaseAttack):

    """The Projected Gradient Descent Attack
    introduced in [1]_ without random start.

    When used without a random start, this attack
    is also known as Basic Iterative Method (BIM)
    or FGSM^k.

    References
    ----------
    .. [1] Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt,
           Dimitris Tsipras, Adrian Vladu, "Towards Deep Learning
           Models Resistant to Adversarial Attacks",
           https://arxiv.org/abs/1706.06083

    .. seealso::

       :class:`LinfinityBasicIterativeAttack` and
       :class:`RandomStartProjectedGradientDescentAttack`

    """

    def _read_dataset(self, fname):
        import pandas as pd
        from sklearn.utils import shuffle

        train_dataframe = pd.read_csv(fname)
        dataframe = shuffle(train_dataframe)

        return dataframe

    # overriden init method in ABC
    def _initialize(self):
        self._remote_model = setup_clf("../model/rf.pkl")
        self._train_data_set = self._read_dataset(
            "../data/hand_preprocessed_trimmed_label_train_gt_augmented_train_set.csv")

    @call_decorator
    def __call__(self, input_or_adv, label=None, unpack=True,
                 binary_search=True,
                 epsilon=0.3,
                 stepsize=0.01,
                 iterations=40,
                 random_start=False,
                 return_early=True,
                 perturbable_idx_set=None,
                 only_increase_idx_set=None,
                 feature_defs=None,
                 normalization_ratios=None,
                 enforce_interval=1,
                 request_id="URL_dummy",
                 model=None,
                 map_back_mode=False,
                 logger=None):
        """Simple iterative gradient-based attack known as
        Basic Iterative Method, Projected Gradient Descent or FGSM^k.

        Parameters
        ----------
        input_or_adv : `numpy.ndarray` or :class:`Adversarial`
            The original, unperturbed input as a `numpy.ndarray` or
            an :class:`Adversarial` instance.
        label : int
            The reference label of the original input. Must be passed
            if `a` is a `numpy.ndarray`, must not be passed if `a` is
            an :class:`Adversarial` instance.
        unpack : bool
            If true, returns the adversarial input, otherwise returns
            the Adversarial object.
        binary_search : bool or int
            Whether to perform a binary search over epsilon and stepsize,
            keeping their ratio constant and using their values to start
            the search. If False, hyperparameters are not optimized.
            Can also be an integer, specifying the number of binary
            search steps (default 20).
        epsilon : float
            Limit on the perturbation size; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        stepsize : float
            Step size for gradient descent; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        iterations : int
            Number of iterations for each gradient descent run.
        random_start : bool
            Start the attack from a random point rather than from the
            original input.
        return_early : bool
            Whether an individual gradient descent run should stop as
            soon as an adversarial is found.
        """

        a = input_or_adv
        del input_or_adv
        del label
        del unpack

        assert epsilon > 0

        # if self._remote_model is None:
        #     self._remote_model = setup_clf("/home/shitong/Desktop/AdGraphAPI/scripts/model/rf.pkl")

        print("Parameters:", epsilon, stepsize, iterations, enforce_interval, request_id)
        self._run(a, binary_search,
                  epsilon, stepsize, iterations,
                  random_start, return_early,
                  perturbable_idx_set, only_increase_idx_set,
                  feature_defs, normalization_ratios,
                  enforce_interval, request_id,
                  model, map_back_mode, logger)


ProjectedGradientDescent = ProjectedGradientDescentAttack
PGD = ProjectedGradientDescent


class RandomStartProjectedGradientDescentAttack(
        LinfinityGradientMixin,
        LinfinityClippingMixin,
        LinfinityDistanceCheckMixin,
        IterativeProjectedGradientBaseAttack):

    """The Projected Gradient Descent Attack
    introduced in [1]_ with random start.

    References
    ----------
    .. [1] Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt,
           Dimitris Tsipras, Adrian Vladu, "Towards Deep Learning
           Models Resistant to Adversarial Attacks",
           https://arxiv.org/abs/1706.06083

    .. seealso:: :class:`ProjectedGradientDescentAttack`

    """

    @call_decorator
    def __call__(self, input_or_adv, label=None, unpack=True,
                 binary_search=True,
                 epsilon=0.3,
                 stepsize=0.01,
                 iterations=40,
                 random_start=True,
                 return_early=True):
        """Simple iterative gradient-based attack known as
        Basic Iterative Method, Projected Gradient Descent or FGSM^k.

        Parameters
        ----------
        input_or_adv : `numpy.ndarray` or :class:`Adversarial`
            The original, unperturbed input as a `numpy.ndarray` or
            an :class:`Adversarial` instance.
        label : int
            The reference label of the original input. Must be passed
            if `a` is a `numpy.ndarray`, must not be passed if `a` is
            an :class:`Adversarial` instance.
        unpack : bool
            If true, returns the adversarial input, otherwise returns
            the Adversarial object.
        binary_search : bool or int
            Whether to perform a binary search over epsilon and stepsize,
            keeping their ratio constant and using their values to start
            the search. If False, hyperparameters are not optimized.
            Can also be an integer, specifying the number of binary
            search steps (default 20).
        epsilon : float
            Limit on the perturbation size; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        stepsize : float
            Step size for gradient descent; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        iterations : int
            Number of iterations for each gradient descent run.
        random_start : bool
            Start the attack from a random point rather than from the
            original input.
        return_early : bool
            Whether an individual gradient descent run should stop as
            soon as an adversarial is found.
        """

        a = input_or_adv
        del input_or_adv
        del label
        del unpack

        assert epsilon > 0

        self._run(a, binary_search,
                  epsilon, stepsize, iterations,
                  random_start, return_early)


RandomProjectedGradientDescent = RandomStartProjectedGradientDescentAttack
RandomPGD = RandomProjectedGradientDescent


class MomentumIterativeAttack(
        LinfinityClippingMixin,
        LinfinityDistanceCheckMixin,
        IterativeProjectedGradientBaseAttack):

    """The Momentum Iterative Method attack
    introduced in [1]_. It's like the Basic
    Iterative Method or Projected Gradient
    Descent except that it uses momentum.

    References
    ----------
    .. [1] Yinpeng Dong, Fangzhou Liao, Tianyu Pang, Hang Su,
           Jun Zhu, Xiaolin Hu, Jianguo Li, "Boosting Adversarial
           Attacks with Momentum",
           https://arxiv.org/abs/1710.06081

    """

    def _gradient(self, a, x, class_, strict=True):
        # get current gradient
        gradient = a.gradient_one(x, class_, strict=strict)
        gradient = gradient / max(1e-12, np.mean(np.abs(gradient)))

        # combine with history of gradient as new history
        self._momentum_history = \
            self._decay_factor * self._momentum_history + gradient

        # use history
        gradient = self._momentum_history
        gradient = np.sign(gradient)
        min_, max_ = a.bounds()
        gradient = (max_ - min_) * gradient
        return gradient

    def _run_one(self, *args, **kwargs):
        # reset momentum history every time we restart
        # gradient descent
        self._momentum_history = 0
        return super(MomentumIterativeAttack, self)._run_one(*args, **kwargs)

    @call_decorator
    def __call__(self, input_or_adv, label=None, unpack=True,
                 binary_search=True,
                 epsilon=0.3,
                 stepsize=0.06,
                 iterations=10,
                 decay_factor=1.0,
                 random_start=False,
                 return_early=True):
        """Momentum-based iterative gradient attack known as
        Momentum Iterative Method.

        Parameters
        ----------
        input_or_adv : `numpy.ndarray` or :class:`Adversarial`
            The original, unperturbed input as a `numpy.ndarray` or
            an :class:`Adversarial` instance.
        label : int
            The reference label of the original input. Must be passed
            if `a` is a `numpy.ndarray`, must not be passed if `a` is
            an :class:`Adversarial` instance.
        unpack : bool
            If true, returns the adversarial input, otherwise returns
            the Adversarial object.
        binary_search : bool
            Whether to perform a binary search over epsilon and stepsize,
            keeping their ratio constant and using their values to start
            the search. If False, hyperparameters are not optimized.
            Can also be an integer, specifying the number of binary
            search steps (default 20).
        epsilon : float
            Limit on the perturbation size; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        stepsize : float
            Step size for gradient descent; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        iterations : int
            Number of iterations for each gradient descent run.
        decay_factor : float
            Decay factor used by the momentum term.
        random_start : bool
            Start the attack from a random point rather than from the
            original input.
        return_early : bool
            Whether an individual gradient descent run should stop as
            soon as an adversarial is found.
        """
        a = input_or_adv
        del input_or_adv
        del label
        del unpack

        assert epsilon > 0

        self._decay_factor = decay_factor

        self._run(a, binary_search,
                  epsilon, stepsize, iterations,
                  random_start, return_early)


MomentumIterativeMethod = MomentumIterativeAttack
