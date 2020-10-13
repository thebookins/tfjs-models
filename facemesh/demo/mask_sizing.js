/*
Functions to size ResMed CPAP masks based on facial measurements

Currently have mask sizing functions for:
- Full face masks: F20, F30, F30i
- Nasal masks: N20, N30i
*/

import { average } from './utils';

// Fit ranges for each mask
var fit_range_F20 = {
    'S': [74,86],
    'M': [86,98],
    'L': [98,110]};

var fit_range_N20 = {
    'S': [23,37],
    'M': [30,45],
    'L': [45,55]};

var fit_range_N30i = {
    'S':  {'NW': [27,40],   'NL': [21.3,33.3]},
    'M':  {'NW': [31,46.5], 'NL': [26,38]},
    'SW': {'NW': [33,50],   'NL': [16.5,32]},
    'W':  {'NW': [43,55],   'NL': [21.3,39]}};

var fit_range_F30 = {
    'S':  {'NW': [26,34], 'NL': [20,33]},
    'M':  {'NW': [34,48], 'NL': [20,38]}};

var fit_range_F30i = {
    'S':  {'NW': [26,40], 'NL': [21,33]},
    'SW': {'NW': [35,49], 'NL': [17,29]},
    'M':  {'NW': [31,45], 'NL': [26,38]},
    'W':  {'NW': [41,55], 'NL': [21,33]}};

// F20 mask
function is_within_bounds_F20(size, faceHeight) {
    let fh_low = fit_range_F20[size][0];
    let fh_upp = fit_range_F20[size][1];
    return faceHeight >= fh_low && faceHeight <= fh_upp;
}

export function mask_sizer_F20(faceHeight) {

    // F20 mask sizing function. Sizing is based on a single measurement, 
    // from the Sellion to Supramenton, only.

    let mask_size = null;
    for (const [key, value] of Object.entries(fit_range_F20)) {
        let low = value[0];
        let upp = value[1];
        if ((faceHeight >= low) && (faceHeight <= upp)) {
            mask_size = key;
        }
    }        
    // If no mask was found, return mask outside range
    if (!mask_size) {
        if (faceHeight < fit_range_F20['S'][0]) {
            mask_size = 'S';
        } else if (faceHeight > fit_range_F20['L'][1]) {
            mask_size = 'L';
        }
    }
    return mask_size;
}

// N20 mask
function is_within_bounds_N20(size, noseBreadth) {
    let nb_low = fit_range_N20[size][0];
    let nb_upp = fit_range_N20[size][1];
    return noseBreadth >= nb_low && noseBreadth <= nb_upp;
}

export function mask_sizer_N20(noseBreadth) {

    // N20 mask sizing function. Sizing is based on noseBreath only.

    let min_score = null;
    let mask_size = null;
    for (const [key, value] of Object.entries(fit_range_N20)) {
        let nb_low = value[0];
        let nb_upp = value[1];
        if ((noseBreadth >= nb_low) && (noseBreadth <= nb_upp)) {
            let nb_centroid = average([nb_low, nb_upp]);
            let score = Math.sqrt(Math.pow((noseBreadth-nb_centroid), 2));
            if ((min_score == null) || (score < min_score)) {
                min_score = score;
                mask_size = key;
            }
        }
    }        
    // If no mask was found, return mask outside range
    if (!mask_size) {
        if (noseBreadth < fit_range_N20['S'][0]) {
            mask_size = 'S';
        } else if (noseBreadth > fit_range_N20['L'][1]) {
            mask_size = 'L';
        }
    }
    return mask_size;
}

// N30i
function is_within_bounds_N30i(size, noseWidth, noseLength) {
    let nw_low = fit_range_N30i[size]['NW'][0];
    let nw_upp = fit_range_N30i[size]['NW'][1];
    let nl_low = fit_range_N30i[size]['NL'][0]; 
    let nl_upp = fit_range_N30i[size]['NL'][1];
    return noseWidth >= nw_low && noseWidth <= nw_upp && noseLength >= nl_low && noseLength <= nl_upp;
}

export function mask_sizer_N30i(noseWidth, noseLength=null, ethnicity=null) {

    // N30i mask sizing function. Sizing is based on noseWidth and noseLength (or ethnicity).

    function mask_size(fit_range, strict=true) {

        let min_score = null;
        let mask_size = null;

        for (const[key, value] of Object.entries(fit_range)) {
            let nw_low = value['NW'][0];
            let nw_upp = value['NW'][1];
            let nl_low = value['NL'][0];
            let nl_upp = value['NL'][1]; 
            let nw_centroid = average([nw_low, nw_upp]);           
            let nl_centroid = average([nl_low, nl_upp]);

            let nw = noseWidth;

            // If noseLength is null, set to the centroid - equivalent to testing for nose width only
            let nl = (noseLength != null) ? noseLength : nl_centroid;

            // If in strict mode, only score the mask if we are within its bounds
            // Otherwise, find the nearest centroid anyway
            if ((!strict) || (nw >= nw_low) && (nw <= nw_upp) && (nl >= nl_low) && (nl <= nl_upp)) {
                let score = Math.sqrt(Math.pow((nw-nw_centroid),2) + Math.pow((nl-nl_centroid),2));
                if ((min_score == null) || (score < min_score)) {
                    min_score = score;
                    mask_size = key;
                }
            }  
        }
        return mask_size;
    }

    // Use noseLength if available, otherwise use ethnicity as a proxy
    // If neither noseLength or ethnicity are available, we can't score so return null
    if (noseLength) {

        let size_strict = mask_size(fit_range_N30i, true);
        let size_notstrict = mask_size(fit_range_N30i, false);
        let size_recommend = (size_strict != null) ? size_strict : size_notstrict;
        return size_recommend;

    } else if (ethnicity) {

        let ethnicities = ['Black/African American', 'East Asian', 'Asian American'];
        let sizes = ethnicities.includes(ethnicity) ? ['SW','W'] : ['S','M'];
        let fit_range = new Object();
        for (const[key, value] of Object.entries(fit_range_N30i)) {
            if (sizes.includes(key)) {
                fit_range[key] = value;
            }
        }

        let size_strict = mask_size(fit_range, true);
        let size_notstrict = mask_size(fit_range, false);
        let size_recommend = (size_strict != null) ? size_strict : size_notstrict;
        return size_recommend;

    } else {

        return null;
    }
}

// F30
function is_within_bounds_F30(size, noseWidth, noseLength) {
    let nw_low = fit_range_F30[size]['NW'][0];
    let nw_upp = fit_range_F30[size]['NW'][1];
    let nl_low = fit_range_F30[size]['NL'][0]; 
    let nl_upp = fit_range_F30[size]['NL'][1];
    return noseWidth >= nw_low && noseWidth <= nw_upp && noseLength >= nl_low && noseLength <= nl_upp;
}

export function mask_sizer_F30(noseWidth, noseLength=null) {

    // F30 mask sizing function. Sizing is based on noseWidth and noseLength.

    function mask_size(fit_range, strict=true) {

        let min_score = null;
        let mask_size = null;

        for (const[key, value] of Object.entries(fit_range)) {
            let nw_low = value['NW'][0];
            let nw_upp = value['NW'][1];
            let nl_low = value['NL'][0];
            let nl_upp = value['NL'][1]; 
            let nw_centroid = average([nw_low, nw_upp]);           
            let nl_centroid = average([nl_low, nl_upp]);

            let nw = noseWidth;

            // If noseLength is null, set to the centroid - equivalent to testing for nose width only
            let nl = (noseLength != null) ? noseLength : nl_centroid;

            // If in strict mode, only score the mask if we are within its bounds
            // Otherwise, find the nearest centroid anyway
            if ((!strict) || (nw >= nw_low) && (nw <= nw_upp) && (nl >= nl_low) && (nl <= nl_upp)) {
                let score = Math.sqrt(Math.pow((nw-nw_centroid),2) + Math.pow((nl-nl_centroid),2));
                if ((min_score == null) || (score < min_score)) {
                    min_score = score;
                    mask_size = key;
                }
            }  
        }
        return mask_size;
    }
    let size_strict = mask_size(fit_range_F30, true);
    let size_notstrict = mask_size(fit_range_F30, false);
    let size_recommend = (size_strict != null) ? size_strict : size_notstrict;
    return size_recommend;
}

// F30i
function is_within_bounds_F30i(size, noseWidth, noseLength) {
    let nw_low = fit_range_F30i[size]['NW'][0];
    let nw_upp = fit_range_F30i[size]['NW'][1];
    let nl_low = fit_range_F30i[size]['NL'][0]; 
    let nl_upp = fit_range_F30i[size]['NL'][1];
    return noseWidth >= nw_low && noseWidth <= nw_upp && noseLength >= nl_low && noseLength <= nl_upp;
}

export function mask_sizer_F30i(noseWidth, noseLength=null, ethnicity=null) {

    // F30i mask sizing function. Sizing is based on noseWidth and noseLength (or ethnicity).

    function mask_size(fit_range, strict=true) {

        let min_score = null;
        let mask_size = null;

        for (const[key, value] of Object.entries(fit_range)) {
            let nw_low = value['NW'][0];
            let nw_upp = value['NW'][1];
            let nl_low = value['NL'][0];
            let nl_upp = value['NL'][1]; 
            let nw_centroid = average([nw_low, nw_upp]);           
            let nl_centroid = average([nl_low, nl_upp]);

            let nw = noseWidth;

            // If noseLength is null, set to the centroid - equivalent to testing for nose width only
            let nl = (noseLength != null) ? noseLength : nl_centroid;

            // If in strict mode, only score the mask if we are within its bounds
            // Otherwise, find the nearest centroid anyway
            if ((!strict) || (nw >= nw_low) && (nw <= nw_upp) && (nl >= nl_low) && (nl <= nl_upp)) {
                let score = Math.sqrt(Math.pow((nw-nw_centroid),2) + Math.pow((nl-nl_centroid),2));
                if ((min_score == null) || (score < min_score)) {
                    min_score = score;
                    mask_size = key;
                }
            }  
        }
        return mask_size;
    }

    // Use noseLength if available, otherwise use ethnicity as a proxy
    // If neither noseLength or ethnicity are available, we can't score so return null
    if (noseLength) {

        let size_strict = mask_size(fit_range_F30i, true);
        let size_notstrict = mask_size(fit_range_F30i, false);
        let size_recommend = (size_strict != null) ? size_strict : size_notstrict;
        return size_recommend;

    } else if (ethnicity) {

        let ethnicities = ['Black/African American', 'East Asian', 'Asian American'];
        let sizes = ethnicities.includes(ethnicity) ? ['SW','W'] : ['S','M'];
        let fit_range = new Object();
        for (const[key, value] of Object.entries(fit_range_F30i)) {
            if (sizes.includes(key)) {
                fit_range[key] = value;
            }
        }

        let size_strict = mask_size(fit_range, true);
        let size_notstrict = mask_size(fit_range, false);
        let size_recommend = (size_strict != null) ? size_strict : size_notstrict;
        return size_recommend;

    } else {

        return null;
    }
}