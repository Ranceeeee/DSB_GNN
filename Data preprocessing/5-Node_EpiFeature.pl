################################ Step 5 #####################################


use strict;

my $cell = "NHEK";
my $resolution = 5000; 
my $interaction_cutoff = 2;

my $dir = "./ChIP-seq_density/";
my $outdir = "./Train_set/Node_EpiFeature_$resolution\_$interaction_cutoff/";
mkdir $outdir unless -e $outdir;

for (my $id = 1;$id<24;$id++)
{
	my $chr = "chr$id";
	$chr = "chrX" if $id == 23;
	open (f1,"$dir/$cell\_$resolution\_$interaction_cutoff.$chr.total_CTCF.density.txt") || die "error";
	open (f2,"$dir/$cell\_$resolution\_$interaction_cutoff.$chr.total_DNase.density.txt") || die "error";
	open (o1,">$outdir/$chr.density.txt") || die "Error";
	$_ = <f1>;
	$_ = <f2>;
	while (<f1>)
	{
		$_=~s/\s+$//;
		my @a = split /\t/,$_;
		my $CTCF = $a[-1];
		$_ = <f2>;
		$_=~s/\s+$//;
		my @a = split /\t/,$_;
		my $DNase = $a[-1];
		print o1 "$DNase\t$CTCF\n";
	}
	close f1;
	close o1;
