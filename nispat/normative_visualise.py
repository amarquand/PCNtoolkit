#!/.../anaconda/bin/python/

from __future__ import print_function
from __future__ import division


"""
* Functions to visualize output of normative modelling
* written by Thomas Wolfers
"""


def vis_zscore_corr(processing_dir,
                    name_txt,
                    thres,
                    outcome_correlates_path,
                    number_of_negative_deviations_path=False,
                    number_of_positive_deviations_path=False,
                    magnitude_of_negative_deviations_path=False,
                    magnitude_of_positive_deviations_path=False,
                    savefigures=True,
                    write_output_totxt=True):
    """
    """
    import seaborn as sns
    from nispat.normative_analyse import zscore_thresholding
    import nispat

    if number_of_negative_deviations_path is False:
        [zscores,
         number_of_positive_deviations,
         number_of_negative_deviations,
         percentage_of_positive_deviations,
         percentage_of_negative_deviations,
         number_of_positive_deviations_forsub,
         number_of_negative_deviations_forsub,
         magnitude_of_positive_deviations_forsub,
         magnitude_of_negative_deviations_forsub,
         percentage_of_positive_deviations_forsub,
         percentage_of_negative_deviations_forsub,
         ] = zscore_thresholding(processing_dir, name_txt, thres, write_output_totxt)
    else:
        number_of_negative_deviations_forsub = nispat.fileio.load_pd(number_of_negative_deviations_path)
        number_of_negative_deviations_forsub = number_of_negative_deviations_forsub[0]
        number_of_positive_deviations_forsub = nispat.fileio.load_pd(number_of_positive_deviations_path)
        number_of_positive_deviations_forsub = number_of_positive_deviations_forsub[0]
        magnitude_of_negative_deviations_forsub = nispat.fileio.load_pd(magnitude_of_negative_deviations_path)
        magnitude_of_negative_deviations_forsub = magnitude_of_negative_deviations_forsub[0]
        magnitude_of_positive_deviations_forsub = nispat.fileio.load_pd(magnitude_of_positive_deviations_path)
        magnitude_of_positive_deviations_forsub = magnitude_of_positive_deviations_forsub[0]


    outcome_correlates = nispat.fileio.load_pd(outcome_correlates_path)
    outcome_correlates=outcome_correlates.fillna(0)
    dim = outcome_correlates.shape

    for x in range(0, dim[1]):
        outcome_correlate = outcome_correlates[x]
        negplot_num = sns.jointplot(number_of_negative_deviations_forsub,
                                outcome_correlate,
                                kind="reg")
        posplot_num = sns.jointplot(number_of_positive_deviations_forsub,
                                outcome_correlate,
                                kind="reg")
        negplot_mag = sns.jointplot(magnitude_of_negative_deviations_forsub,
                                outcome_correlate,
                                kind="reg")
        posplot_mag = sns.jointplot(magnitude_of_positive_deviations_forsub,
                                outcome_correlate,
                                kind="reg")
        if savefigures is True:
            negplot_num.savefig(processing_dir +
                            'number_of_negative_deviations_forsub' +
                            'corr_' +
                            str(x) +
                            '.png')
            posplot_num.savefig(processing_dir +
                            'number_of_positive_deviations_forsub' +
                            'corr_' +
                            str(x) +
                            '.png')
            negplot_mag.savefig(processing_dir +
                            'magnitude_of_negative_deviations_forsub' +
                            'corr_' +
                            str(x) +
                            '.png')
            posplot_mag.savefig(processing_dir +
                            'magnitude_of_positive_deviations_forsub' +
                            'corr_' +
                            str(x) +
                            '.png')
