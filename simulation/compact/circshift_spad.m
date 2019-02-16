function img = circshift_spad(img, z)

for k = 1:size(img,2)
    img(:,k) = circshift(img(:,k),int16(floor(-z(k)))).*(z(k)-floor(z(k))) + ...
               circshift(img(:,k),int16(ceil(-z(k)))).*(ceil(z(k))-z(k));
end