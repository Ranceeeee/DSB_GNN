################################ Step 2 #####################################

use strict;

my $resolution = 5000;
my $interaction_cutoff = 2;

########################## constructing NODE FEATURE MATRIX ######################
use lib qw(BioPerl-1.7.2-2); # NEEDS BIOPERL BioPerl-1.7.2-2
$| = 1;
use strict;
use Getopt::Long;
use File::Spec;
use File::Basename;
use Bio::DB::Fasta;

my $seqs;
print STDERR "reading genome file...\n";
my $GENOMEFILE = "/media/lihao/Data/Genome_bin/hg19/hg19.fa";# input genome fasta
my $seqDb =  Bio::DB::Fasta->new($GENOMEFILE);
my $dir = "./NHEK_interaction_$resolution\_$interaction_cutoff/";
my $outdir = "./Train_set/Node_feature_$resolution\_$interaction_cutoff/"; # output file path
mkdir $outdir unless -e $outdir;

############################## get fasta  ###########################
for (my $i = 1;$i<24;$i++)
{
	my $chr = "chr$i";
	$chr = "chrX" if $i == 23;
	open (f1,"$dir$chr\_5000_node.txt") || die "Error";
	open (o1,">$outdir/$chr.fasta") || die "error";
	while (<f1>)
	{
		$_=~s/\s+$//;
		my @a = split /\t/,$_;
		my ($chr,$start,$end,@rest)=split /\t/,$_;
		next if $chr eq "chrY";
		print o1 ">$a[0]\-$a[1]\-$a[2]\-$a[4]\n";
		#print "$chr\t$start\t$end\n";
		my $Seq = uc($seqDb->get_Seq_by_id($chr)->subseq($start=>$end));
		print o1 "$Seq\n";
	}
	close f1;
	close o1;
}
#########################################################################################

########################### constructing LABEL and ADJANCY MATRIX #########################
my $dir = "./NHEK_interaction_$resolution\_$interaction_cutoff/";
my $seqdir = "./Train_set/Node_feature_$resolution\_$interaction_cutoff/";  
my $labelfile = "./DSB_interaction_$resolution\_$interaction_cutoff/interaction_contain_DSB.txt"; 
my $outdir = "./Train_set/adjancy_matrix_$resolution\_$interaction_cutoff/"; 
mkdir $outdir unless -e $outdir;
my $labeloutdir = "./Train_set/label_$resolution\_$interaction_cutoff/";
mkdir $labeloutdir unless -e $labeloutdir;
my $Nodeoutdir = "./Train_set/Node_$resolution\_$interaction_cutoff/"; 
mkdir $Nodeoutdir unless -e $Nodeoutdir;

################### read LABEL info ##########################
open (f1,"$labelfile") || die "Error";
my %label;
while (<f1>)
{
	$_=~s/\s+$//;
	my @a = split /\t/,$_;
	my $p = join "\t",$a[0],$a[1];
	$label{$p} = 1;
}
close f1;
#######################################################################

my %coding = ("A"=>0,"G"=>1,"C"=>2,"T"=>3);
my $test = "AGCT";
for (my $i = 1;$i<24;$i++)
{
	my $chr = "chr$i";
	$chr = "chrX" if $i == 23;
	print "$chr\n";
	################### converting feature 2 AGCT feature #################
	open (f1,"$seqdir$chr.fasta") || die "Error";
	#open (o1,">$seqdir$chr.seq.coding.txt") || die "Error";
	open (o2,">$seqdir$chr.newfasta") || die "error";
	my %value;
	while (<f1>)
	{
		$_=~s/\s+$//;
		my @a = split />|-/,$_;
		my $p = join "\t",$a[1],$a[2];
		$_=<f1>;
		$_=~s/\s+$//;
		my $seq = $_;
		my $newSeq;
		my $l = length($seq);
		for (my $i = 0;$i<$l;$i++)
		{
			my $c = substr($seq,$i,1);
			next if $coding{$c} eq "";
			if ($newSeq eq ""){$newSeq = $coding{$c};}
			else{$newSeq = join "\t",$newSeq,$coding{$c};}
		}

		my $l = scalar(split /\t/,$newSeq);
		next if $l < $resolution;
		$value{$p} = 1;
		#print o1 "$newSeq\n";
		print o2 "$seq\n";
	}
	close f1;
	#close o1;
	close o2;
	system ("rm $seqdir$chr.fasta");
		
	################### read Node info ###########################
	my @node;
	open (f1,"$dir$chr\_$resolution\_node.txt") || die "Error";
	open (o1,">$labeloutdir/$chr.label.txt") || die "error";
	open (o2,">$Nodeoutdir/$chr.node.txt") || die "Error";
	my $num = 1;
	while (<f1>)
	{
		$_=~s/\s+$//;
		my @a = split /\t/,$_;
		$node[$a[1]] = $num;
		my $tag = 0;
		my $p = join "\t",$a[0],$a[1];
		next if $value{$p} eq "";
		$tag = 1 if $label{$p} == 1;
		print o1 "$num\t$tag\n";
		print o2 "$_\n";
		$num++;
	}
	close f1;
	close o1;
	close o2;
	
	##################### writing ADJANCY MATRIX ##################
	open (f1,"$dir$chr\_$resolution\_KR.txt") || die "Error $dir$chr\_$resolution\_KR.txt";
	open (o1,">$outdir$chr.adjancy.triplet.txt") || die "Error";
	while (<f1>)
	{
		$_=~s/\s+$//;
		my @a = split /\t/,$_;
		my ($p1, $p2) = ( (join "\t",$a[0],$a[1]), (join "\t",$a[0],$a[2]));
		next if $value{$p1} eq "" || $value{$p2} eq "";
		print o1 "$node[$a[1]]\t$node[$a[2]]\t$a[3]\n";
	}
	close f1;
	close o1;
}
