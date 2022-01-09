######################### Step 1 #######################################

use strict;
################################### parameters ###########################
my $cellLine = "NHEK"; 
my $DSB = "./$cellLine/$cellLine.DSB.txt";# input DSB file
my $HICspare_mat_dir = "./$cellLine/$cellLine.hic/";# input KR file of HiC

my $interaction_cutoff = 2;# cutoff for contact ValidatedPaired number 
my $resolution = 5000; # resolution for Hi-C default 5 kb 
my $outdir_HIC_sigInter = "./NHEK_interaction_$resolution\_$interaction_cutoff/";# output path
mkdir $outdir_HIC_sigInter unless -e $outdir_HIC_sigInter;
###############################################################################


############################### for node information #######################
my $interaction_num_total = 0;
my $node_num_total = 0;
my $merge_file;
open (summary,">$outdir_HIC_sigInter/summary_report.txt") || die "Errror";
print "chr\tinteraction num\tnode num\n";
print summary "chr\tinteraction num\tnode num\n";
my @score;
for (my $i = 1;$i<24;$i++)
{
	my $chr = "chr$i";
	$chr = "chrX" if $i == 23;
	open (f1,"$HICspare_mat_dir$chr\_$resolution\_KR.txt") || die "Error";
	open (o1,">$outdir_HIC_sigInter$chr\_$resolution\_KR.txt") || die "Error";
	open (o2,">$outdir_HIC_sigInter$chr\_$resolution\_node.temp") || die "Error";
	my %temp;
	my $interaction_num = 0;
	my $node_num = 0;
	while (<f1>)
	{
		$_=~s/\s+$//;
		my @a = split /\t/,$_;
		push(@score,$a[-1]);
		next if $a[-1] < $interaction_cutoff;
		$interaction_num++;
		print o1 "$chr\t$a[0]\t$a[1]\t$a[2]\n";
		my ($start_left, $end_left) = ($a[0], $a[0]+$resolution-1);
		my ($start_right,$end_right) = ($a[1],$a[1]+$resolution-1);
		my ($p_left, $p_right) = ( (join "\t",$start_left,$end_left), (join "\t",$start_right, $end_right));
		next if $temp{$p_left} ne "";
		print o2 "$chr\t$p_left\n";
		$node_num++;
		$temp{$p_left} = 1;
		next if $temp{$p_right} ne "";
		print o2 "$chr\t$p_right\n";
		$node_num++;
		$temp{$p_right} = 1;
	}
	close f1;
	close o1;
	close o2;
	system ("sort-bed $outdir_HIC_sigInter$chr\_$resolution\_node.temp > $outdir_HIC_sigInter$chr\_$resolution\_node.txt");
	system ("rm $outdir_HIC_sigInter$chr\_$resolution\_node.temp");
	if ($merge_file eq ""){$merge_file = "$outdir_HIC_sigInter$chr\_$resolution\_node.txt";}
	else {$merge_file = join " ",$merge_file,"$outdir_HIC_sigInter$chr\_$resolution\_node.txt";}
	print "$chr\t$interaction_num\t$node_num\n";
	$interaction_num_total+=$interaction_num;
	$node_num_total+=$node_num;
}
print "total interaction num: $interaction_num_total\n";
print "total node num: $node_num_total\n";
print summary "total interaction num: $interaction_num_total\n";
print summary "total node num: $node_num_total\n";
############################################################################################


##################################### for statistics #############################
open (score_output,">$outdir_HIC_sigInter/summary_interaction_score.txt") || die "Errror";
@score = sort{$b <=> $a}@score;
my $l = scalar(@score);
print "score length\t$l\n";
my $interaction_coverage = sprintf("%.1f",10000*$interaction_num_total/$l);
print "interaction coverge: $interaction_coverage*1e-4\n";

my $bin = 100;
my $window = 1/$bin;
my $gap = int($l/$bin);
for (my $i = 0;$i<10;$i++)
{
	my $s = $score[$gap*($i+1)];
	my $w = $window*($i+1)*100;
	print "$w% percentile: $s\n";
	print score_output "$w% percentile: $s\n";
}


my $bin = 20;
my $window = 1/$bin;
my $gap = int($l/$bin);
for (my $i = 1;$i<$bin;$i++)
{
	my $s = $score[$gap*($i+1)];
	my $w = $window*($i+1)*100;
	print "$w% percentile: $s\n";
	print score_output "$w% percentile: $s\n";
}
close score_output;
###################################################################################################



###################################### relationship between Node and DSB ###############################
system ("bedops --merge $merge_file > $outdir_HIC_sigInter/merge\_$resolution\_node.txt");
my $n = count("$outdir_HIC_sigInter/merge\_$resolution\_node.txt");
my $coverage = sprintf("%.2f",$resolution*100*$n/3e9);
print "node genome covreage: $coverage%\n";
print summary "node genome covreage: $coverage%\n";

my $DSB_outdir = "./DSB_interaction_$resolution\_$interaction_cutoff/";
mkdir $DSB_outdir unless -e $DSB_outdir;
system ("bedops -e -1 $DSB $outdir_HIC_sigInter/merge\_$resolution\_node.txt > $DSB_outdir/DSB_in_interaction.txt");
system ("bedops -e -1 $outdir_HIC_sigInter/merge\_$resolution\_node.txt $DSB > $DSB_outdir/interaction_contain_DSB.txt");
my $n = count("$DSB");
print "total DSB: $n\n";
print summary "total DSB: $n\n";
my $n = count("$DSB_outdir/DSB_in_interaction.txt");
print "total DSB in interaction: $n\n";
print summary "total DSB in interaction: $n\n";
my $n = count ("$DSB_outdir/interaction_contain_DSB.txt");
print "Interaction contain DSB: $n\n";
print summary "Interaction contain DSB: $n\n";

close summary;
#########################################################################################################

sub count{
	my @a = @_;
	my $n;
	open (f1,"$a[0]") || die "Error";
	while (<f1>){$n++;}
	close f1;
	return $n;
}
