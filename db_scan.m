function labels = db_scan(data_points, eps, min_points)
labels = zeros(length(data_points),1);
cluster_label = 0;
for i = 1:length(data_points)
    if labels(i) == 0
        neigh_points = get_neigh_points(data_points, i, eps);
        if length(neigh_points) < min_points
            labels(i) = -1;
        else 
            cluster_label = cluster_label + 1;
            labels(i) = cluster_label;
            labels = cluster_expansion(data_points, labels, neigh_points, cluster_label, eps, min_points);
        end
    end
end        
end


function labels = cluster_expansion(data_points, labels, neigh_points, cluster_label, eps, min_points)
j = 1;
while j < length(neigh_points)
    point1 = neigh_points(j);
    if labels(point1) == -1
        labels(point1) = cluster_label;
    elseif labels(point1) == 0
        neigh = get_neigh_points(data_points, point1, eps);
        if length(neigh)>=min_points
            neigh_points = [neigh_points, neigh];
            labels(point1) = cluster_label;
        end
    end
    j = j+1;
end
end


function neigh = get_neigh_points(data_points, point, eps)
neigh = [];
m = length(data_points);
for i = 1:m   
    dist = norm(data_points(point,:)-data_points(i,:));
    if dist < eps
        neigh(end+1) = i;
    end
end
end