################################ Step 4 #####################################

use strict;

my $computeMatrix = "/media/lihao/Data/software/deepTools/bin/computeMatrix";


my $resolution = 5000;
my $interaction_cutoff = 2;

my $cell = "NHEK";
my $dir = "./Train_set/Node_$resolution\_$interaction_cutoff/";

my $dataFile = "./$cell/ChIP-seq/";
my $outdir = "./ChIP-seq_density/";
mkdir $outdir unless -e $outdir;

my $id = $ARGV[0];
my $chr = "chr$id";
$chr = "chrX" if $id == 23;

open (f1,"$dir/$chr.node.txt") || die "ERror";

my $binSize = 100;
my $gap = $resolution/$binSize;

my $regionFile = "$outdir$cell\_$resolution\_$interaction_cutoff.$chr.txt";
my $smallRegionFile = "$outdir$cell\_$resolution\_$interaction_cutoff.$chr\_$binSize\_$gap.txt";
#=cut;
open (o1,">$regionFile") || die "Error";
open (o2,">$smallRegionFile") || die "Error";
while (<f1>)
{
	$_=~s/\s+$//;
	my @a = split /\t/,$_;
	my $end = $a[2] + 1;
	print o1 "$a[0]\t$a[1]\t$end\n";
	for (my $i = 0;$i<$gap;$i++)
	{
		my ($s, $e) = ($a[1]+$binSize*$i, $a[1]+$binSize*($i+1));
		print o2 "$a[0]\t$s\t$e\n";
	}
}
close f1;
close o1;

#=cut;

my $window = $resolution/2;
	print "Total CTCF density\n";
	system ("$computeMatrix scale-regions -m $resolution -bs $resolution -R $regionFile -S $dataFile/$cell\_CTCF.bigWig -o $outdir$cell\_$resolution\_$interaction_cutoff.$chr.total_CTCF.density.gz");
	system ("gunzip -c $outdir$cell\_$resolution\_$interaction_cutoff.$chr.total_CTCF.density.gz > $outdir$cell\_$resolution\_$interaction_cutoff.$chr.total_CTCF.density.txt");

	print "Total DNASE density\n";
	system ("$computeMatrix scale-regions -m $resolution -bs $resolution -R $regionFile -S $dataFile/$cell\_DNase.bigWig -o $outdir$cell\_$resolution\_$interaction_cutoff.$chr.total_DNase.density.gz");
	system ("gunzip -c $outdir$cell\_$resolution\_$interaction_cutoff.$chr.total_DNase.density.gz > $outdir$cell\_$resolution\_$interaction_cutoff.$chr.total_DNase.density.txt");

my $Swindow = $binSize/2;
	print "Small CTCF density\n";
	system ("$computeMatrix scale-regions -m $resolution -bs $binSize -R $smallRegionFile -S $dataFile/$cell\_CTCF.bigWig -o $outdir$cell\_$resolution\_$interaction_cutoff.$chr.small_CTCF.density.gz");
	system ("gunzip -c $outdir$cell\_$resolution\_$interaction_cutoff.$chr.small_CTCF.density.gz > $outdir$cell\_$resolution\_$interaction_cutoff.$chr.small_CTCF.density.txt");
	print "Small DNase density\n";
	system ("$computeMatrix scale-regions -m $resolution -bs $binSize -R $smallRegionFile -S $dataFile/$cell\_DNase.bigWig -o $outdir$cell\_$resolution\_$interaction_cutoff.$chr.small_DNase.density.gz");
	system ("gunzip -c $outdir$cell\_$resolution\_$interaction_cutoff.$chr.small_DNase.density.gz > $outdir$cell\_$resolution\_$interaction_cutoff.$chr.small_DNase.density.txt");
