function env2fish                                                                                                
    if not isatty
        while read line
            set --append argv $line
        end
    end
    string replace -- = ' ' $argv |
        string replace -r -- '^export' --export |
        string replace -r '^(.)' 'set -g $1'
end
