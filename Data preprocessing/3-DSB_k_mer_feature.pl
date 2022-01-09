################################ Step 3 #####################################

use strict;

my $cellLine = "NHEK";
my $resolution = 5000;
my $interaction_cutoff = 2;

my $dir = "./Train_set/Node_feature_$resolution\_$interaction_cutoff/";
my $fasta_dir = "./$cellLine\_fasta/";
mkdir $fasta_dir unless -e $fasta_dir;

for (my $i = 1; $i < 24;$i++)
{
	my $chr = "chr$i";
	$chr = "chrX" if $i == 23;

	for (my $mer = 3;$mer < 6;$mer++)
	{
		my $outdir = "$dir/DNA-seq_mer\_$mer/";
		mkdir $outdir unless -e $outdir;
		open (f1,"$dir$chr.newfasta") || die "Error";
		open (o1,">$outdir/$chr.$mer.txt") || die "Error";
		my $ID = 1;
		my @totalseq;
		for (my $i = 0;$i<$mer;$i++)
		{
			if (scalar(@totalseq) == 0)
			{
				push (@totalseq,"A");
				push (@totalseq,"C");
				push (@totalseq,"G");
				push (@totalseq,"T");
			}
			else
			{
				my @seqtemp;
				foreach my $m(@totalseq)
				{
					push(@seqtemp,join "",$m,"A");
					push(@seqtemp,join "",$m,"C");
					push(@seqtemp,join "",$m,"G");
					push(@seqtemp,join "",$m,"T");
				}
				@totalseq = @seqtemp;
			}
		}
		my @merseq = sort{$a cmp $b}@totalseq;
		my $merl = scalar(@merseq);
		print "$mer mer\t$merl\n";
		
		while (<f1>)
		{
			print "\t$ID\n" if $ID % 10000 == 0;
			$_=~s/\s+$//;
			my $seq = $_;
			my $l = length($seq);
			my %num;
			for (my $i = 0; $i<$l-$mer;$i++)
			{
				my $c = substr($seq,$i,$mer);
				$num{$c}++;
			}
			if ($ID < 2)
			{
				print o1 "seq ID";
				foreach my $m(@merseq)
				{
					print o1 "\t$m";
				}
				print o1 "\n";
			}
			
			print o1 "$ID";
			foreach my $m(@merseq){
				$num{$m} = 0 if $num{$m} eq "";
				print o1 "\t$num{$m}";
			}
			print o1 "\n";
			$ID++;
		}
		close f1;
		close o1;
	}
	system ("mv $dir$chr.newfasta $fasta_dir$chr.newfasta");
}
