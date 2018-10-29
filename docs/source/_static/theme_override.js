$(function() {
    var div_replacer = function(r, hn) {
        var id_str = '', sub_class='rubric-sub';
        if (!hn) {
            hn = 6;
            sub_class = 'rubric-sub rubric-default';
        }
        if ($(r).attr('id')) id_str = 'id="' + $(r).attr('id') + '" ';
        $(r).replaceWith('<div ' + id_str + 'class="rubric"><h' + hn + ' class="' + sub_class + '">'
            + $(r).html() + '</h' + hn + '></div>');
    };
    for (var hn = 1; hn <= 6; hn++) {
        $('.rubric-h' + hn).each(function (i, r) { div_replacer(r, hn); });
    }
    $('p.rubric').each(function (i, r) { div_replacer(r); });
    $('.rubric').each(function (i, r) {
        var rubric_id = 'rubric' + (i + 1);
        if ($(r).attr('id')) {
            rubric_id = $(r).attr('id');
        } else {
            $(r).attr('id', rubric_id);
        }
        $(r).find('.rubric-sub').append('<a class="headerlink" href="#' + rubric_id + '">¶</a>');
    });
});
