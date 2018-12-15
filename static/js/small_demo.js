/**
 * Created by erdem on 12.06.17.
 */

function sendToServer(data, url,  callback) {
    console.log(url)
    console.log(data)
    console.log("=====================")
    $.ajax({
        type: 'POST',
        url: url,
        data: data,
        dataType: 'text',
        success: callback
    });
}


function renderList(resultList) {
    var response = $.parseJSON(resultList);
    $("#results tr").not(function(){ return !!$(this).has('th').length; }).remove();
    $.each(response, function(i, item) {
        $('<tr>')
            .append( $('<td>').text(item.phrase),
                $('<td>').text(item.score))
            .appendTo('#results');
    });
}

$(document).ready(function () {
    $("#send").click(function (e) {
        var req = $("#query").val();
        console.log(req)
        $.get("/api/segment?domain_name="+ req+"&restore_estonian=yes", function (res) {
            res = $.parseJSON(res);
            var splitted = res.splitted_domain_name.map(function (el) {
                return [el]
            });
            var language = res.language;
            sendToServer({data: JSON.stringify(splitted)}, "/api/combinations?language="+ language, renderList);

        });

    });
});