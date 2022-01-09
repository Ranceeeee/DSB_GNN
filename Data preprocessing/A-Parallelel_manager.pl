#use strict;
use File::Basename;
use File::Copy;
use Parallel::ForkManager; 

$nprocs = 30;
$MAX_PROCESSES = $nprocs;
$pm = new Parallel::ForkManager($MAX_PROCESSES);
print STDOUT "Start: An $nprocs parallel manager!\n==================\n";

DATA_LOOP:
for (my $i = 1;$i < 23;$i++)
{
	my $pid = $pm->start and next DATA_LOOP;
	system ("perl 4-ChIP-seq_density.pl $i");
	$pm->finish;
}
$pm->wait_all_children;

