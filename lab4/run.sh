mpicc -o row row.c -lm

echo "_______________"
echo "Nieskalowalny: "
for ((i=1; i<=$1; i++)); do
        mpiexec -machinefile ./allnodes_12 -np $i ./row $2;
done

echo "______________"
echo "Skalowlany:"
for ((j=1; j<=$1; j++)); do
	mpiexec -machinefile ./allnodes_12 -np $j ./row $(($j*$2));
done
