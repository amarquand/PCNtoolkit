#!/.../anaconda/bin/python/

from __future__ import print_function
from __future__ import division

"""
* Functions to analyse output of normative modelling
* written by Thomas Wolfers
"""


def zscore_thresholding(processing_dir,
                        name_txt,
                        thres,
                        write_output_totxt=True):
    """
    """
    # import necessary modules
    import nispat
    import numpy
    zscores = nispat.fileio.load_pd(processing_dir + name_txt)
    dim = zscores.shape

    positive_deviation = zscores[zscores > thres]
    negative_deviation = zscores[zscores < (-thres)]

    #
    pos_num = numpy.nan_to_num(positive_deviation)
    neg_num = numpy.nan_to_num(negative_deviation)

    #
    magnitude_of_positive_deviations_forsub = numpy.sum(pos_num, 0)
    magnitude_of_negative_deviations_forsub = numpy.sum(neg_num, 0)

    #
    pos_num[pos_num > 1] = 1
    neg_num[neg_num < -1] = 1
    number_of_positive_deviations = numpy.sum(pos_num, 1)
    percentage_of_positive_deviations = (number_of_positive_deviations/dim[1])*100
    number_of_negative_deviations = numpy.sum(neg_num, 1)
    percentage_of_negative_deviations = (number_of_negative_deviations/dim[1])*100
    
    number_of_positive_deviations_forsub = numpy.sum(pos_num, 0)
    percentage_of_positive_deviations_forsub = (number_of_positive_deviations_forsub/dim[0])*100
    number_of_negative_deviations_forsub = numpy.sum(neg_num, 0)
    percentage_of_negative_deviations_forsub = (number_of_negative_deviations_forsub/dim[0])*100

    #
    if write_output_totxt is True:
        numpy.savetxt(processing_dir + name_txt +
                      '_number_of_positive_deviations.txt',
                      number_of_positive_deviations,
                      fmt='%1.10e',
                      delimiter=' ')
        numpy.savetxt(processing_dir + name_txt +
                      '_number_of_negative_deviations.txt',
                      number_of_negative_deviations,
                      fmt='%1.10e',
                      delimiter=' ')
        numpy.savetxt(processing_dir + name_txt +
                      'number_of_positive_deviations_forsub.txt',
                      number_of_positive_deviations_forsub,
                      fmt='%1.10e',
                      delimiter=' ')
        numpy.savetxt(processing_dir + name_txt +
                      'number_of_negative_deviations_forsub.txt',
                      number_of_negative_deviations_forsub,
                      fmt='%1.10e',
                      delimiter=' ')
        numpy.savetxt(processing_dir + name_txt +
                      'percentage_of_positive_deviations.txt',
                      percentage_of_positive_deviations,
                      fmt='%1.10e',
                      delimiter=' ')
        numpy.savetxt(processing_dir + name_txt +
                      'percentage_of_negative_deviations.txt',
                      percentage_of_negative_deviations,
                      fmt='%1.10e',
                      delimiter=' ')
        numpy.savetxt(processing_dir + name_txt +
                      'magnitude_of_positive_deviations_forsub.txt',
                      magnitude_of_positive_deviations_forsub,
                      fmt='%1.10e',
                      delimiter=' ')
        numpy.savetxt(processing_dir + name_txt +
                      'magnitude_of_negative_deviations_forsub.txt',
                      magnitude_of_negative_deviations_forsub,
                      fmt='%1.10e',
                      delimiter=' ')
        numpy.savetxt(processing_dir + name_txt +
                      'percentage_of_positive_deviations_forsub.txt',
                      percentage_of_positive_deviations_forsub,
                      fmt='%1.10e',
                      delimiter=' ')
        numpy.savetxt(processing_dir + name_txt +
                      'percentage_of_negative_deviations_forsub.txt',
                      percentage_of_negative_deviations_forsub,
                      fmt='%1.10e',
                      delimiter=' ')

    return(zscores,
           number_of_positive_deviations,
           number_of_negative_deviations,
           percentage_of_positive_deviations,
           percentage_of_negative_deviations,
           number_of_positive_deviations_forsub,
           number_of_negative_deviations_forsub,
           magnitude_of_positive_deviations_forsub,
           magnitude_of_negative_deviations_forsub,
           percentage_of_positive_deviations_forsub,
           percentage_of_negative_deviations_forsub)
