function y = pre_emphasis(x, alpha)
    if nargin < 2
        alpha = 0.97;
    end
    y = filter([1, -alpha], 1, x);
end
