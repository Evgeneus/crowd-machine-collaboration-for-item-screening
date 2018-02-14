from collections import defaultdict


class CrowdVoting:

    def __init__(self, filename):
        self.filename = filename
        self.votes_per_paper = defaultdict(list)
        self.agg_vote_per_cr = [{} for _ in range(4)]
        self.power_per_cr = [{} for _ in range(4)]



    def get_votes_proportion(self):
        lines=[]
        with open(self.filename) as f:

            lines = f.read().splitlines()

        for item in lines:

            info=item.split('\t')
            self.votes_per_paper[info[7]].append(info[9] +','+ info[10] + ',' +info[11] +','+info[12])


    def get_crowd_votes(self):


        for k in self.votes_per_paper.keys():
            cr_all_votes = [0] * 4
            cr_pos_votes = [0] * 4

            for p in range(5):

                    if (self.votes_per_paper[k][p]).split(',')[0] !='-':
                        cr_all_votes[0]+=1

                    if (self.votes_per_paper[k][p]).split(',')[1] != '-':
                        cr_all_votes[1]+=1

                    if (self.votes_per_paper[k][p]).split(',')[2] != '-':
                        cr_all_votes[2]+=1


                    if (self.votes_per_paper[k][p]).split(',')[3]!='-':
                        cr_all_votes[3]+=1

            for p in range(5):

                    if (self.votes_per_paper[k][p]).split(',')[0] == "Yes" or (self.votes_per_paper[k][p]).split(',')[0] =="CantTell":
                        cr_pos_votes[0] += 1

                    if (self.votes_per_paper[k][p]).split(',')[1] == "Yes" or (self.votes_per_paper[k][p]).split(',')[1] == "CantTell":
                        cr_pos_votes[1]+=1

                    if (self.votes_per_paper[k][p]).split(',')[2] == "Yes" or (self.votes_per_paper[k][p]).split(',')[2] == "CantTell":
                        cr_pos_votes[2]+=1

                    if (self.votes_per_paper[k][p]).split(',')[3]!='-' and (self.votes_per_paper[k][p]).split(',')[3]!="NoInfo":

                            if int((self.votes_per_paper[k][p]).split(',')[3])>= 10:
                                cr_pos_votes[3]+=1


            for i in range(4):
                if cr_all_votes[i] > 0:
                    is_mv = (cr_all_votes[i] / 2) + 1
                    if cr_pos_votes[i] >=is_mv:
                        self.agg_vote_per_cr[i][k] = 1
                    else:
                        self.agg_vote_per_cr[i][k] = 0

                    self.power_per_cr[i][k] = (cr_all_votes[i]-cr_pos_votes[i])/float(cr_all_votes[i])


