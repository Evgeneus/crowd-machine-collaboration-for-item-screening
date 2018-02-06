from realdata_experiment.crowd_voting import CrowdVoting


def crowd_main(criteria):
    cp=CrowdVoting("../data/ProtonBeamCrowddata.txt")
    cp.get_votes_proportion()
    cp.get_crowd_votes()
    return cp.agg_vote_per_cr[criteria]